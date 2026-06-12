import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: THE REGIME HUNT
# Extracts streaming features and classifies market regimes (Normal, Fat Finger,
# Sector Rotation, Correlation Break) using O(n^2) rolling covariance updates.
# ─────────────────────────────────────────────────────────────────────────────

class RollingCovarianceEngine:
    def __init__(self, n_assets, window):
        self.n = n_assets
        self.w = window
        self.buf = np.zeros((window, n_assets), dtype=np.float64)
        self.ptr = 0          
        self.count = 0        


        self.sum_x = np.zeros(n_assets, dtype=np.float64)    
        self.sum_xx = np.zeros((n_assets, n_assets), dtype=np.float64)

    def push(self, r):
        r = r.astype(np.float64)

        if self.count < self.w:
            self.buf[self.ptr] = r
            self.sum_x  += r
            self.sum_xx += np.outer(r, r)
            self.ptr = (self.ptr + 1) % self.w
            self.count += 1
            return None
        else:
            r_old = self.buf[self.ptr]
            self.sum_x  += r - r_old
            self.sum_xx += np.outer(r, r) - np.outer(r_old, r_old)
            self.buf[self.ptr] = r
            self.ptr = (self.ptr + 1) % self.w
            return self._covariance()

    def _covariance(self):
        mu = self.sum_x / self.w
        cov_raw = (self.sum_xx - self.w * np.outer(mu, mu)) / (self.w - 1)
        return 0.5 * (cov_raw + cov_raw.T)

def fast_subspace_angle_deg(V1, V2):
    M = V1 @ V2.T          
    _, sv, _ = np.linalg.svd(M, full_matrices=False)
    angles_rad = np.arccos(np.clip(sv, -1.0, 1.0))
    return float(np.degrees(angles_rad).mean())

def vectorized_predict(features_df, ff_thresh, cr_thresh, sr_thresh, q_thresh):
    ff  = features_df["max_contrib_share"].values
    ang = features_df["subspace_angle_deg"].values
    ev1 = features_df["ev_ratio_pc1_delta"].values
    q   = features_df["q_stat"].values

    labels = np.full(len(features_df), "NORMAL", dtype=object)

    labels[ev1 > sr_thresh]                              = "SECTOR_ROTATION"
    labels[ang > cr_thresh]                              = "CORRELATION_BREAK"
    labels[(ff > ff_thresh) & (q > q_thresh)]            = "FAT_FINGER"

    nan_mask = np.isnan(q)
    labels[nan_mask] = "NORMAL"

    return pd.Series(labels, index=features_df.index, dtype=str)

def vectorized_smooth(labels_series, smoothing=5):
    REGIME_MAP = {
        "NORMAL": 0, "SECTOR_ROTATION": 1,
        "CORRELATION_BREAK": 2, "FAT_FINGER": 3
    }
    REGIME_INV = {v: k for k, v in REGIME_MAP.items()}

    raw = np.array([REGIME_MAP[x] for x in labels_series], dtype=np.int8)
    n = len(raw)
    out = np.zeros(n, dtype=np.int8)
    out[0] = raw[0]

    if smoothing > 1 and n >= smoothing:
        padded = np.concatenate([
            np.full(smoothing - 1, raw[0], dtype=np.int8), raw
        ])
        shape   = (n, smoothing)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                                  strides=strides)
        uniform = (windows == windows[:, -1:]).all(axis=1)
        committed = raw.copy()
        committed[~uniform] = -1    

        current = raw[0]
        for i in range(n):
            if committed[i] != -1:
                current = committed[i]
            out[i] = current
    else:
        out[:] = raw

    return pd.Series(
        [REGIME_INV[x] for x in out],
        index=labels_series.index,
        dtype=str
    )

def build_streaming_features(returns_df, window,
                              monday_eigvecs_k, monday_eigvecs_2,
                              train_mean, monday_ev_ratios, q_threshold):
    data = np.ascontiguousarray(returns_df.values, dtype=np.float64)
    n_ticks, n_assets = data.shape

    I_minus_P = np.eye(n_assets) - monday_eigvecs_k.T @ monday_eigvecs_k
    V_mon = np.ascontiguousarray(monday_eigvecs_2)  

    feat_names = [
        "ev_ratio_pc1", "ev_ratio_pc2", "mean_corr",
        "q_stat", "subspace_angle_deg",
        "max_contrib_share", "ev_ratio_pc1_delta",
    ]
    out = np.full((n_ticks, len(feat_names)), np.nan)

    engine = RollingCovarianceEngine(n_assets, window)

    for t in range(n_ticks):
        r = data[t]
        cov = engine.push(r)

        if cov is None:
            continue    

        vals = np.linalg.eigvalsh(cov)[::-1]   
        total_var = vals.sum()
        if total_var < 1e-15:
            continue
        ev1 = vals[0] / total_var
        ev2 = vals[1] / total_var

        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-10, 1e-10, std)
        corr = cov / np.outer(std, std)
        idx_u = np.triu_indices(n_assets, k=1)
        mean_corr = corr[idx_u].mean()

        r_c = r - train_mean
        residual = I_minus_P @ r_c
        q_t = float(np.dot(residual, residual))
        contrib = residual ** 2
        max_share = contrib.max() / (q_t + 1e-15)

        _, vecs = np.linalg.eigh(cov)
        V_roll = np.ascontiguousarray(vecs[:, -2:].T)   
        angle_deg = fast_subspace_angle_deg(V_mon, V_roll)

        ev1_delta = abs(ev1 - monday_ev_ratios[0])

        out[t] = [ev1, ev2, mean_corr, q_t, angle_deg, max_share, ev1_delta]

    return pd.DataFrame(out, index=returns_df.index, columns=feat_names)

class RegimeClassifier:
    def __init__(self, ff_thresh, cr_thresh, sr_thresh, q_thresh, smoothing=5):
        self.ff_thresh  = ff_thresh
        self.cr_thresh  = cr_thresh
        self.sr_thresh  = sr_thresh
        self.q_thresh   = q_thresh
        self.smoothing  = smoothing

    def predict_batch(self, features_df):
        raw = vectorized_predict(
            features_df,
            self.ff_thresh, self.cr_thresh, self.sr_thresh, self.q_thresh
        )
        return vectorized_smooth(raw, self.smoothing)

    def predict_single(self, feat_row_np):
        if np.any(np.isnan(feat_row_np)):
            return "NORMAL"

        _, _, _, q_t, ang, ff, ev1_d = feat_row_np

        if ff > self.ff_thresh and q_t > self.q_thresh:
            return "FAT_FINGER"
        if ang > self.cr_thresh:
            return "CORRELATION_BREAK"
        if ev1_d > self.sr_thresh:
            return "SECTOR_ROTATION"
        return "NORMAL"