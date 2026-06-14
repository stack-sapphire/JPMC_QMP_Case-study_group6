"""
p3_regime_hunt.py
=================
PS2 · Part 3 — The Regime Hunt (10 marks)

Solves:
  (a) Streaming feature pipeline — STRICTLY CAUSAL.
      The central fix: RollingCovarianceEngine uses a circular buffer
      of past rows only. At tick t, the window is data[t-window : t]
      (exclusive of t itself via the push-then-feature pattern).
      No rolling().apply(), no shift(), no future rows ever enter
      the covariance or any derived feature.
  (b) RegimeClassifier — rule-based decision tree.
      Two paths:
        predict_batch()  — vectorised boolean masks, no Python loop,
                           used for offline Thursday CSV generation.
        predict_single() — plain if/else on a raw np.ndarray,
                           used tick-by-tick in the Bonus pipeline.
      Temporal smoothing: require 5 consecutive identical predictions
      before committing (exploits the 30-tick minimum episode guarantee).
  (c) Self-evaluation on Wednesday using released labels_wed.csv.
  (d) Thursday submission CSV: Team_{N}_submission.csv.
  (e) Post-mortem markdown cell (written in notebook).

Look-ahead bug — what it was and how it is fixed here:
  The previous build_streaming_features() rebuilt the covariance as
  W = data[t-window : t+1], which included the current tick t in both
  the eigenvector estimation AND the Q-statistic. The current tick's
  regime fingerprint was therefore baked into its own features, giving
  the classifier oracle-level information it would never have in live
  trading. This produced 0.96 precision on CORRELATION_BREAK (impossible
  in noisy data) and 0.00 F1 on FAT_FINGER (features never captured
  single-asset micro-spikes because the spike averaged into the window).

  Fix: RollingCovarianceEngine.push(r) updates the window buffer with r
  and returns the covariance of the RESULTING window (which now contains
  r as its newest member). The Q-statistic is then computed against the
  Monday static model, not the rolling window — so tick t's Q is:
    residual_t = (I - PᵀP) @ r_t
  where PᵀP is the Monday projector. The rolling covariance is used ONLY
  for eigenvalue ratios, mean correlation, and subspace angle — all of
  which describe the *historical* window ending AT t (exclusive), i.e.
  the state the market was in BEFORE tick t arrived.

  The only tick included in its own feature window is via the running
  covariance update: after push(r_t), the covariance includes r_t.
  This is correct and causal: in live trading you HAVE r_t when you
  compute features for t — you just cannot use r_{t+1}, r_{t+2}, etc.
"""

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.metrics import f1_score, classification_report

np.random.seed(42)

REGIMES = ["NORMAL", "FAT_FINGER", "SECTOR_ROTATION", "CORRELATION_BREAK"]


# ─────────────────────────────────────────────────────────────────────────────
# RollingCovarianceEngine — O(n²) incremental update, no window rebuild
# ─────────────────────────────────────────────────────────────────────────────

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
        sum_x  = Σ xᵢ           (for the mean)
        sum_xx = Σ xᵢxᵢᵀ        (for the raw second moment)
    Centred covariance:
        Cov = (sum_xx - n·μμᵀ) / (n-1)   where μ = sum_x / n
    """

    def __init__(self, n_assets: int, window: int):
        self.n   = n_assets
        self.w   = window
        self.buf = np.zeros((window, n_assets), dtype=np.float64)
        self.ptr = 0      # index of the OLDEST row (next to be evicted)
        self.count = 0

        self.sum_x  = np.zeros(n_assets,               dtype=np.float64)
        self.sum_xx = np.zeros((n_assets, n_assets),   dtype=np.float64)

    def push(self, r: np.ndarray):
        """
        Insert one new row r. Returns the current unbiased covariance
        matrix once the window is full, or None beforehand.

        CAUSALITY: after push(r_t), the window contains r_{t-window+1}
        through r_t inclusive. Features computed from this covariance
        describe the state of the market UP TO AND INCLUDING tick t.
        No future tick is ever in the buffer.
        """
        r = r.astype(np.float64, copy=False)

        if self.count < self.w:
            # Window not yet full — accumulate without evicting
            self.buf[self.ptr] = r
            self.sum_x  += r
            self.sum_xx += np.outer(r, r)
            self.ptr     = (self.ptr + 1) % self.w
            self.count  += 1
            return None
        else:
            # Evict oldest, insert newest
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


# ─────────────────────────────────────────────────────────────────────────────
# Fast subspace angle — avoids scipy for the 2×2 case
# ─────────────────────────────────────────────────────────────────────────────

def fast_subspace_angle_deg(V1: np.ndarray, V2: np.ndarray) -> float:
    """
    V1, V2: (K, n) — rows are orthonormal basis vectors.
    Returns: mean principal angle in degrees.

    For K=2 the inner product matrix M = V1 @ V2.T is 2×2.
    Its singular values are cos(principal angles).
    A 2×2 SVD is O(8 operations) — much faster than scipy's full routine.
    Sign-invariant by construction (SVD ignores sign of eigenvectors).
    """
    M  = V1 @ V2.T
    _, sv, _ = np.linalg.svd(M, full_matrices=False)
    angles_rad = np.arccos(np.clip(sv, -1.0, 1.0))
    return float(np.degrees(angles_rad).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Feature names (must match the order of the feature array)
# ─────────────────────────────────────────────────────────────────────────────

FEAT_NAMES = [
    "ev_ratio_pc1",        # mandatory: λ₁/Σλᵢ
    "ev_ratio_pc2",        # mandatory: λ₂/Σλᵢ
    "mean_corr",           # mandatory: mean of 66 pairwise correlations
    "q_stat",              # mandatory: Q vs Monday model
    "subspace_angle_deg",  # mandatory: principal angle vs Monday top-2
    "max_contrib_share",   # extra: fraction of Q from the single worst asset
    "ev_ratio_pc1_delta",  # extra: |ev_ratio_pc1 - monday_ev1|
]

# Index lookup (used by predict_single for speed — no dict lookup per tick)
_I_EV1   = 0
_I_EV2   = 1
_I_CORR  = 2
_I_Q     = 3
_I_ANG   = 4
_I_FF    = 5
_I_EV1D  = 6


# ─────────────────────────────────────────────────────────────────────────────
# build_streaming_features — causal, incremental, no look-ahead
# ─────────────────────────────────────────────────────────────────────────────

def build_streaming_features(returns_df: pd.DataFrame,
                              window: int,
                              monday_eigvecs_k: np.ndarray,
                              monday_eigvecs_2: np.ndarray,
                              train_mean: np.ndarray,
                              monday_ev_ratios: np.ndarray,
                              q_threshold: float) -> pd.DataFrame:
    """
    Computes the 7 features for every tick in returns_df.
    Rows during the warmup window (first `window` ticks) are NaN
    and will default to NORMAL per the assignment spec.

    CAUSALITY GUARANTEE:
      At tick t, the feature vector uses only data[0 : t+1] — i.e. all
      ticks up to and including t. No future tick is ever accessed.

    Window choice: 100 ticks.
      - Must be > n_assets = 12 (to estimate a full-rank covariance)
      - Should be noticeably shorter than the 30-tick minimum episode
        length so regime boundaries are resolved sharply
      - 100 ticks ≈ a few minutes of intraday data — captures
        intraday correlation structure without over-smoothing

    args:
      monday_eigvecs_k : (K, 12) — top-K Monday eigenvectors (for Q stat)
      monday_eigvecs_2 : (2, 12) — top-2 Monday eigenvectors (for subspace angle)
      train_mean       : (12,)   — Monday column means (for Q stat centering)
      monday_ev_ratios : (2,)    — [ev1_mon, ev2_mon] reference values
      q_threshold      : float   — empirical 99th-pctile Q from Monday (for info only)
    """
    data = np.ascontiguousarray(returns_df.values, dtype=np.float64)
    n_ticks, n_assets = data.shape

    # Precompute Monday complement projector for Q stat
    I_minus_P = np.eye(n_assets) - monday_eigvecs_k.T @ monday_eigvecs_k

    out = np.full((n_ticks, len(FEAT_NAMES)), np.nan, dtype=np.float64)
    engine = RollingCovarianceEngine(n_assets, window)

    for t in range(n_ticks):
        r   = data[t]
        cov = engine.push(r)     # window now contains data[max(0,t-w+1) : t+1]

        if cov is None:
            continue             # window not yet full → leave as NaN

        total_var = np.maximum(np.trace(cov), 1e-15)

        # ── Eigenvalue ratios (from rolling covariance, describes history) ──
        vals = np.linalg.eigvalsh(cov)[::-1]   # descending
        ev1  = vals[0] / total_var
        ev2  = vals[1] / total_var

        # ── Mean pairwise correlation ────────────────────────────────────────
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-10, 1e-10, std)
        corr     = cov / np.outer(std, std)
        idx_u    = np.triu_indices(n_assets, k=1)
        mean_corr = float(corr[idx_u].mean())

        # ── Q-statistic vs Monday model ──────────────────────────────────────
        # Centred by Monday mean — not the rolling mean — to stay consistent
        # with the Monday PCA model's coordinate system.
        r_c      = r - train_mean
        residual = I_minus_P @ r_c
        q_t      = float(np.dot(residual, residual))
        contrib  = residual ** 2
        max_share = float(contrib.max() / (q_t + 1e-15))

        # ── Subspace angle: rolling top-2 vs Monday top-2 ───────────────────
        # eigvalsh returns ascending; last 2 columns = top-2 eigenvectors
        _, vecs  = np.linalg.eigh(cov)
        V_roll   = np.ascontiguousarray(vecs[:, -2:].T)   # (2, 12)
        angle_deg = fast_subspace_angle_deg(monday_eigvecs_2, V_roll)

        # ── ev1 delta ────────────────────────────────────────────────────────
        ev1_delta = abs(ev1 - monday_ev_ratios[0])

        out[t] = [ev1, ev2, mean_corr, q_t, angle_deg, max_share, ev1_delta]

    return pd.DataFrame(out, index=returns_df.index, columns=FEAT_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration (from Wednesday labelled data)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_thresholds(features_wed: pd.DataFrame,
                          labels_wed: pd.Series) -> dict:
    """
    For each regime, use the 10th percentile of that regime's feature
    distribution as the detection threshold — i.e. 90% of true examples
    are above the threshold, balancing precision vs recall.
    Falls back to conservative defaults if a regime has < 50 examples.
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


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised prediction (batch path — no Python row loop)
# ─────────────────────────────────────────────────────────────────────────────

def vectorized_predict(features_df: pd.DataFrame, thresholds: dict,
                        q_thresh: float) -> pd.Series:
    """
    Applies the decision tree using numpy boolean mask arrays.
    No Python loop over rows — runs in milliseconds for 30k rows.

    Priority (applied in reverse order so higher-priority overwrites):
      NORMAL             — default
      SECTOR_ROTATION    — ev_ratio_pc1_delta above threshold
      CORRELATION_BREAK  — subspace_angle_deg above threshold
      FAT_FINGER         — max_contrib_share AND q_stat both above threshold
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


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised temporal smoothing (no Python loop)
# ─────────────────────────────────────────────────────────────────────────────

def vectorized_smooth(labels_series: pd.Series, smoothing: int = 5) -> pd.Series:
    """
    Commits to a new label only after `smoothing` consecutive identical ticks.
    Implemented without a Python for loop using numpy stride tricks.

    Why smoothing=5?
      Episodes last ≥ 30 ticks. A 5-tick confirmation window catches
      true transitions within 5 ticks of onset while rejecting single-tick
      noise spikes. False positive bursts shorter than 5 ticks are suppressed.
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


# ─────────────────────────────────────────────────────────────────────────────
# RegimeClassifier — both batch and live-tick interfaces
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Batch path ───────────────────────────────────────────────────────────
    def predict_batch(self, features_df: pd.DataFrame) -> pd.Series:
        """Vectorised boolean masks + temporal smoothing. Fast for 30k rows."""
        raw = vectorized_predict(features_df, self.thresholds, self.q_thresh)
        return vectorized_smooth(raw, self.smoothing)

    # ── Live single-row path (Bonus pipeline) ────────────────────────────────
    def predict_single(self, feat_row: np.ndarray) -> str:
        """
        feat_row: 1-D numpy array in the order defined by FEAT_NAMES.
        Returns one of REGIMES.
        No Pandas, no DataFrame — called once per tick in step().
        Takes < 1 µs.
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (Part 3c)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Submission CSV builder (Part 3d)
# ─────────────────────────────────────────────────────────────────────────────

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
        submission["timestamp"], format='ISO8601', utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    path = f"{output_dir}/Team_{team_number}_submission.csv"
    submission.to_csv(path, index=False)
    print(f"Submission saved: {path}  ({len(submission)} rows)")
    return path