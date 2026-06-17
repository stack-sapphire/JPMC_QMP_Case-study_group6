"""
bonus_pipeline.py

Full end-to-end integration of the Phantom Lattice architecture. 

The main engine is `PhantomLatticePipeline.step(r)`, which takes a single 
aligned tick (12-asset return vector) and runs it through the entire stack:
1. PCA Detector (extracts Q and T^2 stats)
2. Regime Classifier (Rolling Covariance updates)
3. Inventory Simulator (checks against dynamic barriers)

It emits clean telemetry during the stream:
- ALERT: Rising edges only for threshold breaches.
- REGIME_CHANGE: Only logs when the market state physically transitions.
- ABSORBED: Fires when inventory hits a barrier.
- SUMMARY: Final aggregate stats at the end of the run.

See the bottom cell for a live 60-minute Thursday simulation.
"""

import gc
import json
import time
import numpy as np
import pandas as pd

# Internal imports from the same src/ folder
from p1_blind_cartographer import load_and_pivot, fit_pca_on_monday
from p2_sentinel_detector import StreamingPCADetector
from p3_regime_classifier import (RollingCovarianceEngine, FEAT_NAMES,
                             fast_subspace_angle_deg, RegimeClassifier,
                             _I_Q, _I_ANG, _I_FF, _I_EV1D)
from p4_model_shootout import fixed_barrier_fn, anomaly_driven_barrier_fn


class PhantomLatticePipeline:
    """
End-to-end live trading risk system. 
Processes streaming market data one 12-asset tick at a time via `step()`.

Args:
    calibration_returns: Monday's (T, 12) returns matrix for baseline state.
    K: Selected number of PCA components.
    q_threshold, t2_threshold: 99th percentile alert limits from Monday.
    clf_thresholds: Dict of regime thresholds (from Part 3).
    train_mean, monday_eigvecs_2, monday_ev_ratios: Monday baseline stats.
    U, L: Standard barrier bounds (default: 40).
    U_tight: Restrictive barrier used during alerts (default: 20).
    target_round_trips: Required T value from Part 4 (default: 10).
    window: Lookback size for rolling covariance features (default: 100).
"""

    def __init__(self,
                 calibration_returns: np.ndarray,
                 K: int,
                 q_threshold: float,
                 t2_threshold: float,
                 clf_thresholds: dict,
                 train_mean: np.ndarray,
                 monday_eigvecs_2: np.ndarray,
                 monday_ev_ratios: np.ndarray,
                 U: int = 40,
                 L: int = 40,
                 U_tight: int = 20,
                 target_round_trips: int = 10,
                 window: int = 100):

        self.K              = K
        self.q_thresh       = q_threshold
        self.t2_thresh      = t2_threshold
        self.train_mean     = train_mean
        self.monday_eigvecs_2 = monday_eigvecs_2
        self.monday_ev_ratios = monday_ev_ratios
        self.U              = U
        self.L              = L
        self.U_tight        = U_tight
        self.T              = target_round_trips
        self.window         = window
        self.n_assets       = calibration_returns.shape[1]

        #  Component 1: fit PCA on Monday calibration data 
        pca_mon = fit_pca_on_monday(
            pd.DataFrame(calibration_returns)
        )
        self.eigvecs_k = np.ascontiguousarray(pca_mon.components_[:K])
        self.eigvals_k = pca_mon.explained_variance_[:K]

        #  Component 2: streaming anomaly detector 
        self.detector = StreamingPCADetector(
            n_assets=self.n_assets, K=K, lam=0.01, warmup=500, cadence=10
        )
        self.detector.warmup_jit()

        # Component 3: rolling covariance + regime classifier 
        self.cov_engine = RollingCovarianceEngine(self.n_assets, window)
        I_minus_P = np.eye(self.n_assets) - self.eigvecs_k.T @ self.eigvecs_k
        self.I_minus_P = I_minus_P

        self.clf = RegimeClassifier(
            thresholds=clf_thresholds,
            q_thresh=q_threshold,
            smoothing=5
        )
        # Simple smoothing buffer for live path
        self._regime_buffer = ["NORMAL"] * 5
        self._current_regime = "NORMAL"

        # Component 4: inventory state 
        self.inventory    = 0
        self.round_trips  = 0
        self.absorbed     = False
        self.max_inv      = 0
        self.max_inv_tick = 0

        #  Telemetry 
        self.telemetry    = []
        self.tick_count   = 0

        # Alert rising-edge state
        self._in_alert    = False
        self._prev_regime = "NORMAL"

        # Disable GC for the hot path
        gc.disable()
        gc.freeze()

    # Core step  one 12-vector tick

    def step(self, r: np.ndarray) -> dict:
        """
Process a single tick.

Args:
    r (np.array): A 12-element array of the current returns. 
                  Make sure this is already pivoted, cleaned (no NaNs), 
                  and aligned exactly with Monday's column order.

Returns:
    dict: A snapshot of the current pipeline state. Note: If the system 
          hits a barrier (absorbed), it freezes and just returns the 
          terminal state for all future ticks.
"""
        if self.absorbed:
            return self._frozen_snapshot()

        t = self.tick_count
        r = np.asarray(r, dtype=np.float64)

        # 1. Centre by Monday mean 
        r_c = r - self.train_mean

        # 2. Streaming anomaly detector 
        q, t2 = self.detector.update(r_c)

        # Rising-edge alert logic (only fire on the first crossing)
        alert_now = (not np.isnan(q) and
                     (q > self.q_thresh or t2 > self.t2_thresh))
        if alert_now and not self._in_alert:
            self._log_event("ALERT", tick=t, q=round(float(q), 6),
                            t2=round(float(t2), 6))
        self._in_alert = alert_now

        # 3. Regime classification 
        # Compute features inline (same logic as build_streaming_features
        # but for a single row -no DataFrame, no Pandas)
        feat_row = self._compute_features(r, r_c, t)
        regime   = self.clf.predict_single(feat_row)

        # Simple majority smoothing over last 5 ticks
        self._regime_buffer.append(regime)
        self._regime_buffer.pop(0)
        smoothed = max(set(self._regime_buffer), key=self._regime_buffer.count)

        if smoothed != self._prev_regime:
            self._log_event("REGIME_CHANGE", tick=t,
                            from_regime=self._prev_regime, to_regime=smoothed)
            self._prev_regime = smoothed
        self._current_regime = smoothed

        #  4. Inventory / kill-switch 
        # Canonical fill construction: sign of net imbalance across 12 assets
        fill = int(np.sign(np.sign(r).sum()))

        if fill != 0:
            # Dynamic barriers: tight if in alert
            barrier = self.U_tight if alert_now else self.U

            self.inventory += fill

            if abs(self.inventory) >= barrier:
                self.absorbed = True
                self._log_event("ABSORBED", tick=t,
                                inventory=self.inventory,
                                round_trips=self.round_trips)

            elif self.inventory == 0:
                self.round_trips += 1
                if self.round_trips >= self.T:
                    self._log_event("TARGET_MET", tick=t)

        # Track inventory peak
        if abs(self.inventory) > self.max_inv:
            self.max_inv      = abs(self.inventory)
            self.max_inv_tick = t

        self.tick_count += 1

        return {
            "tick":         t,
            "q":            None if np.isnan(q) else round(float(q), 6),
            "t2":           None if np.isnan(t2) else round(float(t2), 6),
            "alert":        alert_now,
            "regime":       self._current_regime,
            "inventory":    self.inventory,
            "round_trips":  self.round_trips,
            "absorbed":     self.absorbed,
        }

    # Feature computation for a single row (no Pandas, no DataFrames)

    def _compute_features(self, r: np.ndarray,
                           r_c: np.ndarray,
                           t: int) -> np.ndarray:
        """
        Computes the 7-element feature vector for the live step() path.
        Uses the same logic as build_streaming_features() but on a 1-D array.
        Returns NaN array during the covariance warmup window.
        """
        feat = np.full(len(FEAT_NAMES), np.nan, dtype=np.float64)

        cov = self.cov_engine.push(r)
        if cov is None:
            return feat   

        total_var = max(np.trace(cov), 1e-15)
        vals = np.linalg.eigvalsh(cov)[::-1]
        ev1  = vals[0] / total_var
        ev2  = vals[1] / total_var

        std  = np.sqrt(np.diag(cov))
        std  = np.where(std < 1e-10, 1e-10, std)
        corr = cov / np.outer(std, std)
        idx_u = np.triu_indices(self.n_assets, k=1)
        mean_corr = float(corr[idx_u].mean())

        residual  = self.I_minus_P @ r_c
        q_t       = float(np.dot(residual, residual))
        contrib   = residual ** 2
        max_share = float(contrib.max() / (q_t + 1e-15))

        _, vecs = np.linalg.eigh(cov)
        V_roll  = np.ascontiguousarray(vecs[:, -2:].T)
        angle   = fast_subspace_angle_deg(self.monday_eigvecs_2, V_roll)

        ev1_delta = abs(ev1 - self.monday_ev_ratios[0])

        feat[:] = [ev1, ev2, mean_corr, q_t, angle, max_share, ev1_delta]
        return feat

    # Telemetry
    
    def _log_event(self, event_type: str, **kwargs):
        self.telemetry.append({"event": event_type, **kwargs})

    def emit_telemetry(self) -> list:
        """Return the telemetry log, appending a SUMMARY entry."""
        summary = {
            "event":         "SUMMARY",
            "total_ticks":   self.tick_count,
            "max_inventory": self.max_inv,
            "max_inv_tick":  self.max_inv_tick,
            "round_trips":   self.round_trips,
            "absorbed":      self.absorbed,
            "verdict":       "RUIN" if self.absorbed else "SURVIVED",
        }
        return self.telemetry + [summary]

    def _frozen_snapshot(self) -> dict:
        return {
            "tick":        self.tick_count,
            "q":           None, "t2": None,
            "alert":       False,
            "regime":      self._current_regime,
            "inventory":   self.inventory,
            "round_trips": self.round_trips,
            "absorbed":    True,
        }

    def __del__(self):
        gc.enable()


# Live demo runner — called from the  LIVE DEMO  notebook cell

def run_live_demo(pipe: PhantomLatticePipeline,
                  returns_thu: pd.DataFrame,
                  chunk_size: int = 100):
    """
    Streams Thursday's first 60 minutes through the pipeline in chunks.
    Prints a summary snapshot after each chunk.
    Must complete in < 60 seconds wall-clock.
    """
    t0            = returns_thu.index.min()
    first_hour    = returns_thu.loc[returns_thu.index < t0 + pd.Timedelta("1H")]
    chunks        = np.array_split(first_hour.values,
                                   max(1, len(first_hour) // chunk_size))

    print(f"Streaming {len(first_hour)} ticks in {len(chunks)} chunks of ~{chunk_size}")
    wall_start = time.perf_counter()

    for chunk_idx, chunk in enumerate(chunks):
        for r in chunk:
            snapshot = pipe.step(r)

        # Print one-line summary per chunk (not per tick — avoids I/O bottleneck)
        print(f"  chunk {chunk_idx+1:3d}/{len(chunks)} | "
              f"tick {snapshot['tick']:5d} | "
              f"regime={snapshot['regime']:<20s} | "
              f"inv={snapshot['inventory']:+3d} | "
              f"trips={snapshot['round_trips']} | "
              f"alert={'YES' if snapshot['alert'] else 'no '} | "
              f"absorbed={'ABSORBED' if snapshot['absorbed'] else 'ok'}")

        if snapshot["absorbed"]:
            print("  *** KILL SWITCH TRIGGERED ***")
            break

    elapsed = time.perf_counter() - wall_start
    print(f"\nCompleted in {elapsed:.2f}s")
    assert elapsed < 60, f"Demo exceeded 60-second wall-clock limit ({elapsed:.1f}s)"

    print("\nTelemetry log:")
    print(json.dumps(pipe.emit_telemetry(), indent=2, default=str))