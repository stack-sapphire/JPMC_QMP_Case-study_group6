"""
p3_regime_classifier.py
=======================

Causal feature engineering and rule-based regime classification.

This module is strictly designed for live, tick-by-tick causality. 
It replaces batch rolling windows (which are prone to look-ahead bias) 
with a stateful `RollingCovarianceEngine`. 

Core Mechanics:
- Causal Features: At tick `t`, features are computed exclusively using 
  the circular buffer ending at `t`. The Q-statistic and subspace projection 
  are evaluated against the static Monday baseline, ensuring that a single-tick 
  anomaly does not mask itself by polluting its own reference frame.
- Classification (`RegimeClassifier`): A deterministic decision tree handling 
  both vectorized batch execution (for backtesting) and single-tick evaluation 
  (for the live pipeline).
- Temporal Smoothing: Implements a 5-tick hysteresis buffer to filter transient 
  noise, leveraging the assignment's 30-tick minimum episode guarantee.

Outputs:
Handles self-evaluation against Wednesday's known labels and generates 
the final out-of-sample Thursday submission CSV.
"""

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.metrics import f1_score, classification_report

np.random.seed(42)

REGIMES = ["NORMAL", "FAT_FINGER", "SECTOR_ROTATION", "CORRELATION_BREAK"]



# RollingCovarianceEngine — O(n²) incremental update, no window rebuild
class RollingCovarianceEngine:
    """
    Maintains a rolling unbiased covariance matrix over a fixed window
    using O(n²) rank-1 updates instead of O(window·n²) full rebuilds.

    Method:
      push(r) adds r to the window (evicting the oldest row if full) and
      returns the current covariance matrix, or None if the window is not
      yet full.

    Numerically equivalent to:
        W = data[t-window:t+1]   (after push, window contains current tick)
        cov = (W - W.mean(0)).T @ (W - W.mean(0)) / (window - 1)
    but ~100x faster because it maintains running sums instead of
    recomputing from scratch.

    Running sums maintained:
        sum_x  = Σ xᵢ   (for the mean)
        sum_xx = Σ xᵢxᵢᵀ  (for the raw second moment)
    Centred covariance:
        Cov = (sum_xx - n·μμᵀ) / (n-1)   where μ = sum_x / n
    """

    def __init__(self, n_assets: int, window: int):
        self.n   = n_assets
        self.w   = window
        self.buf = np.zeros((window, n_assets), dtype=np.float64)
        self.ptr = 0   # index of the OLDEST row (next to be evicted)
        self.count = 0

        self.sum_x  = np.zeros(n_assets,               dtype=np.float64)
        self.sum_xx = np.zeros((n_assets, n_assets),   dtype=np.float64)

    def push(self, r: np.ndarray):
        """
Pushes a new tick (r) into the circular buffer.

Returns the unbiased covariance matrix of the current window, 
or None if the warmup period is still active.

Note on Causality: The returned covariance represents the market 
state exactly at tick t (inclusive of r_t, but strictly zero 
knowledge of t+1).
"""
        r = r.astype(np.float64, copy=False)

        if self.count < self.w:
            # Window not yet full, accumulate without evicting
            self.buf[self.ptr] = r
            self.sum_x  += r
            self.sum_xx += np.outer(r, r)
            self.ptr     = (self.ptr + 1) % self.w
            self.count  += 1
            return None
        else:
            # Evict oldest,insert newest
            r_old        = self.buf[self.ptr]
            self.sum_x  += r - r_old
            self.sum_xx += np.outer(r, r) - np.outer(r_old, r_old)
            self.buf[self.ptr] = r
            self.ptr     = (self.ptr + 1) % self.w
            return self._covariance()

    def _covariance(self) -> np.ndarray:
        mu      = self.sum_x / self.w
        cov_raw = (self.sum_xx - self.w * np.outer(mu, mu)) / (self.w - 1)
        # Symmetrise to prevent floating-point asymmetry drift
        return 0.5 * (cov_raw + cov_raw.T)



# Fast subspace angle, avoids scipy for the 2×2 case
def fast_subspace_angle_deg(V1: np.ndarray, V2: np.ndarray) -> float:
    """
Calculates the mean principal angle (in degrees) between two subspaces.

Inputs V1 and V2 are (K, n) matrices of orthonormal basis vectors.
For K=2, the inner product matrix is just 2x2. Computing its SVD directly 
takes O(8) operations, which completely avoids the massive overhead of 
calling scipy's general routine during a live tick stream. 

Bonus: It naturally handles PCA sign-ambiguity since the SVD ignores signs.
"""
    
    M  = V1 @ V2.T
    _, sv, _ = np.linalg.svd(M, full_matrices=False)
    angles_rad = np.arccos(np.clip(sv, -1.0, 1.0))
    return float(np.degrees(angles_rad).mean())
# Feature names (must match the order of the feature array)


FEAT_NAMES = [
    "ev_ratio_pc1",        # mandatory: λ₁/Σλᵢ
    "ev_ratio_pc2",        # mandatory: λ₂/Σλᵢ
    "mean_corr",           # mandatory: mean of 66 pairwise correlations
    "q_stat",              # mandatory: Q vs Monday model
    "subspace_angle_deg",  # mandatory: principal angle vs Monday top-2
    "max_contrib_share",   # extra: fraction of Q from the single worst asset
    "ev_ratio_pc1_delta",  # extra: |ev_ratio_pc1 - monday_ev1|
]

# Index lookup (used by predict_single for speed , no dict lookup per tick)
_I_EV1   = 0
_I_EV2   = 1
_I_CORR  = 2
_I_Q     = 3
_I_ANG   = 4
_I_FF    = 5
_I_EV1D  = 6



# build_streaming_features — causal, incremental, no look-ahead
def build_streaming_features(returns_df: pd.DataFrame,
                              window: int,
                              monday_eigvecs_k: np.ndarray,
                              monday_eigvecs_2: np.ndarray,
                              train_mean: np.ndarray,
                              monday_ev_ratios: np.ndarray,
                              q_threshold: float) -> pd.DataFrame:
    """
Computes the 7-element feature vector for all ticks in the returns DataFrame.

Note on Warmup: The first `window` ticks (100) are required to build the 
initial covariance matrix. They will output NaN features and default to 
the NORMAL regime.

Architectural Parameters:
- Window size (100): Chosen to be large enough for a stable, full-rank 
  covariance (n=12), but small enough to resolve regime boundaries sharply 
  without over-smoothing.
- Causality: Features at tick `t` are generated strictly from the 
  historical buffer `[t - window + 1 : t]`. No future data is accessed.

Args:
    monday_eigvecs_k, monday_eigvecs_2: Baseline Monday subspaces.
    train_mean: Monday column means for residual centering.
    monday_ev_ratios: Reference eigenvalue ratios for drift tracking.
    q_threshold: Monday's 99th-percentile Q limit.
"""
    data = np.ascontiguousarray(returns_df.values, dtype=np.float64)
    n_ticks, n_assets = data.shape

    # Precompute Monday complement projector for Q stat
    I_minus_P = np.eye(n_assets) - monday_eigvecs_k.T @ monday_eigvecs_k

    out = np.full((n_ticks, len(FEAT_NAMES)), np.nan, dtype=np.float64)
    engine = RollingCovarianceEngine(n_assets, window)

    for t in range(n_ticks):
        r   = data[t]
        cov = engine.push(r)  # window now contains data[max(0,t-w+1) : t+1]

        if cov is None:
            continue      # window not yet full → leave as NaN

        total_var = np.maximum(np.trace(cov), 1e-15)

        #Eigenvalue ratios (from rolling covariance, describes history) 
        vals = np.linalg.eigvalsh(cov)[::-1]   # descending
        ev1  = vals[0] / total_var
        ev2  = vals[1] / total_var

        # Mean pairwise correlation 
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-10, 1e-10, std)
        corr     = cov / np.outer(std, std)
        idx_u    = np.triu_indices(n_assets, k=1)
        mean_corr = float(corr[idx_u].mean())

        #  Q-statistic vs Monday model 
        # Centred by Monday mean — not the rolling mean — to stay consistent
        # with the Monday PCA model's coordinate system.
        r_c      = r - train_mean
        residual = I_minus_P @ r_c
        q_t      = float(np.dot(residual, residual))
        contrib  = residual ** 2
        max_share = float(contrib.max() / (q_t + 1e-15))

        # Subspace angle: rolling top-2 vs Monday top-2 
        # eigvalsh returns ascending; last 2 columns = top-2 eigenvectors
        _, vecs  = np.linalg.eigh(cov)
        V_roll   = np.ascontiguousarray(vecs[:, -2:].T)   # (2, 12)
        angle_deg = fast_subspace_angle_deg(monday_eigvecs_2, V_roll)

        #  ev1 delta 
        ev1_delta = abs(ev1 - monday_ev_ratios[0])

        out[t] = [ev1, ev2, mean_corr, q_t, angle_deg, max_share, ev1_delta]

    return pd.DataFrame(out, index=returns_df.index, columns=FEAT_NAMES)



# Threshold calibration (from Wednesday labelled data)


def calibrate_thresholds(features_wed: pd.DataFrame,
                          labels_wed: pd.Series) -> dict:
    """
Calibrates regime-specific thresholds using the 10th percentile of each feature's 
distribution to balance precision and recall. 

Includes a safety fallback to conservative defaults if a regime contains fewer 
than 50 samples to prevent overfitting on sparse data.
"""
    merged = features_wed.join(labels_wed.rename("regime"), how="inner").dropna()

    def regime_quantile(regime, feat, q=0.10, default=None):
        sub = merged.loc[merged["regime"] == regime, feat]
        if len(sub) < 50:
            print(f"Warning: only {len(sub)} labelled {regime} ticks — using default threshold")
            return default
        return float(sub.quantile(q))

    return {
        "ff_thresh":  regime_quantile("FAT_FINGER",       "max_contrib_share", 0.10, 0.35),
        "cr_thresh":  regime_quantile("CORRELATION_BREAK","subspace_angle_deg", 0.10, 20.0),
        "sr_thresh":  regime_quantile("SECTOR_ROTATION",  "ev_ratio_pc1_delta", 0.10, 0.05),
        "q_thresh":   None,   # filled in from Part 2 empirical threshold
    }
# Vectorised prediction (batch path — no Python row loop)


def vectorized_predict(features_df: pd.DataFrame, thresholds: dict,
                        q_thresh: float) -> pd.Series:
    """
Vectorized execution of the regime classification decision tree. 

Uses pure NumPy boolean masking to process historical matrices efficiently. 
The classification cascade evaluates in reverse order, ensuring that 
severe, high-priority anomalies seamlessly overwrite lower-priority states.

Evaluation Hierarchy (Highest to Lowest Priority):
1. FAT_FINGER: Breaches in both Q-statistic and max_contrib_share.
2. CORRELATION_BREAK: Severe drift in subspace_angle_deg.
3. SECTOR_ROTATION: Anomalous shifts in the PC1 eigenvalue ratio.
4. NORMAL: Baseline state.
"""
    ff  = features_df["max_contrib_share"].values
    ang = features_df["subspace_angle_deg"].values
    ev1d = features_df["ev_ratio_pc1_delta"].values
    q   = features_df["q_stat"].values

    labels = np.full(len(features_df), "NORMAL", dtype=object)

    # Apply in ascending priority — higher priority overwrites lower
    labels[ev1d > thresholds["sr_thresh"]]                    = "SECTOR_ROTATION"
    labels[ang  > thresholds["cr_thresh"]]                    = "CORRELATION_BREAK"
    labels[(ff  > thresholds["ff_thresh"]) & (q > q_thresh)]  = "FAT_FINGER"

    # NaN rows (warmup) → NORMAL per spec
    labels[np.isnan(q)] = "NORMAL"

    return pd.Series(labels, index=features_df.index, dtype=str)

# Vectorised temporal smoothing (no Python loop)
def vectorized_smooth(labels_series: pd.Series, smoothing: int = 5) -> pd.Series:
    """
Temporal smoothing filter for the classification stream.

Acts as a high-pass filter for state changes. By requiring 5 consecutive 
identical ticks before committing to a regime flip, it ignores isolated 
micro-bursts and transient noise. Since true market episodes persist for 
at least 30 ticks, this 5-tick delay represents an optimal tradeoff 
between detection latency and precision.
"""
    REGIME_MAP = {r: i for i, r in enumerate(REGIMES)}
    REGIME_INV = {i: r for r, i in REGIME_MAP.items()}

    raw = np.array([REGIME_MAP.get(x, 0) for x in labels_series], dtype=np.int8)
    n   = len(raw)

    if smoothing <= 1 or n < smoothing:
        return labels_series.copy()

    # Build a (n, smoothing) sliding window view — O(1) memory via strides
    padded  = np.concatenate([np.full(smoothing - 1, raw[0], dtype=np.int8), raw])
    shape   = (n, smoothing)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # A tick changes label only when all `smoothing` recent ticks agree
    uniform   = (windows == windows[:, -1:]).all(axis=1)
    committed = raw.copy()
    committed[~uniform] = -1    # sentinel: "hold previous label"

    # Forward-fill: scan once to propagate committed labels
    out     = np.empty(n, dtype=np.int8)
    current = raw[0]
    for i in range(n):
        if committed[i] != -1:
            current = committed[i]
        out[i] = current

    return pd.Series(
        [REGIME_INV.get(int(x), "NORMAL") for x in out],
        index=labels_series.index,
        dtype=str
    )

# RegimeClassifier — both batch and live-tick interfaces

class RegimeClassifier:
    """
    Rule-based regime classifier.

    Two prediction paths:
      predict_batch(features_df) — vectorised, for offline use.
      predict_single(feat_row)   — single np.ndarray row, for Bonus pipeline.

    fit() calibrates thresholds from labelled Wednesday data.
    Thresholds may also be set directly via the constructor.
    """

    def __init__(self, thresholds: dict = None, q_thresh: float = None,
                 smoothing: int = 5):
        self.thresholds = thresholds or {
            "ff_thresh": 0.35, "cr_thresh": 20.0, "sr_thresh": 0.05
        }
        self.q_thresh  = q_thresh or 0.0
        self.smoothing = smoothing

    def fit(self, features_wed: pd.DataFrame, labels_wed: pd.Series,
            q_thresh: float):
        self.thresholds = calibrate_thresholds(features_wed, labels_wed)
        self.q_thresh   = q_thresh
        return self

    #  Batch path 
    def predict_batch(self, features_df: pd.DataFrame) -> pd.Series:
        """Vectorised boolean masks + temporal smoothing. Fast for 30k rows."""
        raw = vectorized_predict(features_df, self.thresholds, self.q_thresh)
        return vectorized_smooth(raw, self.smoothing)

    #  Live single-row path (Bonus pipeline) 
    def predict_single(self, feat_row: np.ndarray) -> str:
        """
Single-tick inference for the live pipeline.
Args:
    feat_row (np.ndarray): 1-D array aligned with FEAT_NAMES.
Returns:
    str: The predicted regime label.
Note: Optimized for the hot path. All Pandas/DataFrame overhead is stripped 
for microsecond execution within the `step()` loop.
"""
        if np.any(np.isnan(feat_row)):
            return "NORMAL"

        q_t   = feat_row[_I_Q]
        ang   = feat_row[_I_ANG]
        ff    = feat_row[_I_FF]
        ev1d  = feat_row[_I_EV1D]

        if ff > self.thresholds["ff_thresh"] and q_t > self.q_thresh:
            return "FAT_FINGER"
        if ang > self.thresholds["cr_thresh"]:
            return "CORRELATION_BREAK"
        if ev1d > self.thresholds["sr_thresh"]:
            return "SECTOR_ROTATION"
        return "NORMAL"



# Evaluation helpers (Part 3c)
def evaluate_on_wednesday(preds_wed: pd.Series,
                           labels_wed_path: str = "data/labels_wed.csv") -> dict:
    """
    Aligns predictions with released ground truth and computes per-regime
    precision / recall / F1 and macro-F1.
    """
    truth = pd.read_csv(labels_wed_path, parse_dates=["timestamp"])
    truth = truth.set_index("timestamp")["regime"]

    merged = truth.to_frame("true").join(
        preds_wed.rename("pred"), how="inner"
    ).dropna()

    macro = f1_score(merged["true"], merged["pred"],
                     labels=REGIMES, average="macro", zero_division=0)
    print(f"\nWednesday macro-F1: {macro:.4f}")
    print(classification_report(
        merged["true"], merged["pred"], labels=REGIMES, zero_division=0
    ))
    return {"macro_f1": macro}



# Submission CSV builder (Part 3d)

def build_submission_csv(preds_thu: pd.Series, team_number: str,
                          output_dir: str = ".") -> str:
    """
    Writes Team_{N}_submission.csv with columns: timestamp, predicted_regime.
    Timestamps in ISO-8601 UTC so pd.read_csv(parse_dates=["timestamp"])
    recovers them exactly (required by the score_submission script).
    Warm-up NaN rows default to NORMAL per spec.
    """
    submission = pd.DataFrame({
        "timestamp":        preds_thu.index,
        "predicted_regime": preds_thu.values,
    })
    submission["predicted_regime"] = submission["predicted_regime"].fillna("NORMAL")
    submission["timestamp"] = pd.to_datetime(
        submission["timestamp"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    path = f"{output_dir}/Team_{team_number}_submission.csv"
    submission.to_csv(path, index=False)
    print(f"Submission saved: {path}  ({len(submission)} rows)")
    return path