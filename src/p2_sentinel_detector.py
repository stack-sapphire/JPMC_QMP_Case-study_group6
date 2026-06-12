import numpy as np
import time
import gc
from numba import njit

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: THE SENTINEL DETECTOR
# Ultra-low latency streaming PCA engine. Uses Numba JIT compilation,
# static memory preallocation, and a fixed cadence to process ticks < 50µs.
# ─────────────────────────────────────────────────────────────────────────────

@njit(fastmath=True, nogil=True, cache=True)
def _hot_tick(r, Sigma, I_minus_P, V, inv_lambda, lam):
    Sigma[:] = (1.0 - lam) * Sigma + lam * np.outer(r, r)
    residual = I_minus_P @ r
    q = np.dot(residual, residual)
    scores = V @ r
    t2 = np.dot(scores * scores, inv_lambda)
    return q, t2

class StreamingPCADetector:
    def __init__(self, n_assets, K, lam=0.01, warmup=500, cadence=10):
        self.K = K
        self.lam = lam
        self.warmup = warmup
        self.cadence = cadence          
        self.n = n_assets
        self.t = 0
        self.ticks_since_eigh = 0
        
        self.Sigma = np.zeros((n_assets, n_assets), dtype=np.float64)
        self.V = np.zeros((K, n_assets), dtype=np.float64)
        self.inv_lambda = np.ones(K, dtype=np.float64)
        self.P_proj = np.zeros((n_assets, n_assets), dtype=np.float64)
        self.I_minus_P = np.eye(n_assets, dtype=np.float64)
        
        self._MAX = 40_000
        self.q_out = np.full(self._MAX, np.nan)
        self.t2_out = np.full(self._MAX, np.nan)
        self.latencies_ns = np.zeros(self._MAX, dtype=np.int64)
        
    def _update_eigenvectors(self):
        vals, vecs = np.linalg.eigh(self.Sigma)
        idx = np.argsort(vals)[::-1][:self.K]
        self.V[:] = np.ascontiguousarray(vecs[:, idx].T)
        self.inv_lambda[:] = 1.0 / np.maximum(vals[idx], 1e-12)  
        self.P_proj[:] = self.V.T @ self.V
        self.I_minus_P[:] = np.eye(self.n) - self.P_proj
        
    def warmup_jit(self):
        dummy = np.zeros(self.n, dtype=np.float64)
        _hot_tick(dummy, self.Sigma.copy(), self.I_minus_P.copy(),
                  self.V.copy(), self.inv_lambda.copy(), self.lam)
        
    def update(self, r):
        t_start = time.perf_counter_ns()
        
        if self.t >= self.warmup:
            self.ticks_since_eigh += 1
            if self.ticks_since_eigh >= self.cadence:
                self._update_eigenvectors()
                self.ticks_since_eigh = 0
        
        q, t2 = _hot_tick(
            r, self.Sigma, self.I_minus_P, self.V, self.inv_lambda, self.lam
        )
        
        t_end = time.perf_counter_ns()
        self.latencies_ns[self.t] = t_end - t_start
        
        if self.t >= self.warmup:
            self.q_out[self.t] = q
            self.t2_out[self.t] = t2
        
        self.t += 1
        return (q, t2) if self.t > self.warmup else (np.nan, np.nan)

    def latency_report(self):
        valid = self.latencies_ns[self.warmup:self.t]
        return {
            "mean_us":   round(valid.mean() / 1000, 2),
            "p99_us":    round(np.percentile(valid, 99) / 1000, 2),
            "p999_us":   round(np.percentile(valid, 99.9) / 1000, 2),
            "max_us":    round(valid.max() / 1000, 2),
        }