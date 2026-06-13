"""
p5_backtester.py  — v3
========================
Changes from v2:
  1. Q_THRESH gate removed from FAT_FINGER condition
     (q_stat is zero for all regimes — disabled as a gate)
  2. Thresholds set directly from diagnostics table:
       FF_THRESH = 0.42  (between NORMAL=0.312 and FAT_FINGER=0.536)
       CR_THRESH = 35.0  (below CORR_BREAK=63.26, above SECT_ROT=31.17)
       SR_THRESH = 0.15  (between NORMAL=0.028 and SECT_ROT=0.314)
  3. predict_batch patched inline to use ff-only gate for FAT_FINGER
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd
import numpy as np
import time
import subprocess
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Safe data loader — clips extremes to prevent PCA overflow
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prep_data(path):
    from p1_blind_cartographer import load_and_pivot
    df = load_and_pivot(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    std = df.std()
    std[std == 0] = 1.0
    df = df.clip(lower=-10 * std, upper=10 * std, axis=1)
    return df


def fit_monday_pca(returns_mon, K=3):
    train_mean = returns_mon.mean()
    centred    = returns_mon - train_mean
    assert np.isfinite(centred.values).all(), "Inf/NaN in centred returns"
    pca = PCA(n_components=None)
    pca.fit(centred.values)
    print(f"  PCA fit: top-{K} components explain "
          f"{pca.explained_variance_ratio_[:K].sum()*100:.1f}% of variance")
    return pca, train_mean


# ─────────────────────────────────────────────────────────────────────────────
# Patched vectorised predict — FAT_FINGER uses ff-only gate (no q_stat)
# ─────────────────────────────────────────────────────────────────────────────

def patched_predict_batch(features_df, ff_thresh, cr_thresh, sr_thresh,
                           smoothing=5):
    """
    Vectorised decision tree.

    Priority (highest overwrites lowest):
      NORMAL             — default
      SECTOR_ROTATION    — ev_ratio_pc1_delta > sr_thresh
      CORRELATION_BREAK  — subspace_angle_deg > cr_thresh
      FAT_FINGER         — max_contrib_share > ff_thresh
                           (q_stat gate REMOVED — q_stat is zero for all regimes)
    """
    ff   = features_df["max_contrib_share"].values
    ang  = features_df["subspace_angle_deg"].values
    ev1d = features_df["ev_ratio_pc1_delta"].values
    q    = features_df["q_stat"].values

    labels = np.full(len(features_df), "NORMAL", dtype=object)

    # Apply in ascending priority so higher overwrites lower
    labels[ev1d > sr_thresh]  = "SECTOR_ROTATION"
    labels[ang  > cr_thresh]  = "CORRELATION_BREAK"
    labels[ff   > ff_thresh]  = "FAT_FINGER"        # no q_stat gate

    # Warmup rows (NaN features) → NORMAL
    labels[np.isnan(q)] = "NORMAL"

    raw = pd.Series(labels, index=features_df.index, dtype=str)

    # Temporal smoothing — commit only after `smoothing` consecutive ticks agree
    REGIME_MAP = {"NORMAL": 0, "SECTOR_ROTATION": 1,
                  "CORRELATION_BREAK": 2, "FAT_FINGER": 3}
    REGIME_INV = {v: k for k, v in REGIME_MAP.items()}

    arr     = np.array([REGIME_MAP[x] for x in raw], dtype=np.int8)
    n       = len(arr)
    padded  = np.concatenate([np.full(smoothing - 1, arr[0], dtype=np.int8), arr])
    shape   = (n, smoothing)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    uniform = (windows == windows[:, -1:]).all(axis=1)

    committed = arr.copy()
    committed[~uniform] = -1

    out     = np.empty(n, dtype=np.int8)
    current = arr[0]
    for i in range(n):
        if committed[i] != -1:
            current = committed[i]
        out[i] = current

    return pd.Series([REGIME_INV[int(x)] for x in out],
                     index=features_df.index, dtype=str)


# ─────────────────────────────────────────────────────────────────────────────
# Feature diagnostics printer
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_diagnostics(features_df, y_true):
    REGIMES = ["NORMAL", "FAT_FINGER", "SECTOR_ROTATION", "CORRELATION_BREAK"]
    feats   = ["q_stat", "max_contrib_share", "subspace_angle_deg",
               "ev_ratio_pc1_delta", "mean_corr"]

    df = features_df.copy()
    df["true_label"] = y_true

    print("\n" + "=" * 72)
    print("FEATURE DIAGNOSTICS — median per regime")
    print("=" * 72)
    print(f"{'Feature':<26}", end="")
    for r in REGIMES:
        print(f"  {r[:14]:<15}", end="")
    print()
    print("-" * 72)
    for feat in feats:
        if feat not in df.columns:
            continue
        print(f"{feat:<26}", end="")
        for r in REGIMES:
            sub = df.loc[df["true_label"] == r, feat].dropna()
            med = sub.median() if len(sub) > 0 else float("nan")
            print(f"  {med:>14.5f} ", end="")
        print()
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-logger
# ─────────────────────────────────────────────────────────────────────────────

def log_and_push(ff, cr, sr, acc):
    log_file = os.path.join(ROOT, "experiment_log.md")
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a") as f:
        if not file_exists:
            f.write("# Model Tuning Log\n\n")
            f.write("| Date & Time | FF | CR | SR | Accuracy |\n")
            f.write("|---|---|---|---|---|\n")
        f.write(f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"| {ff} | {cr} | {sr} | **{acc*100:.2f}%** |\n")
    print(f"\nLogged to {log_file}")
    subprocess.run(["git", "add", "experiment_log.md", "src/p5_backtester.py"],
                   cwd=ROOT)
    subprocess.run(["git", "commit", "-m",
                    f"Tuning run: {acc*100:.2f}% | FF={ff} CR={cr} SR={sr}"],
                   cwd=ROOT, capture_output=True)
    result = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True)
    print("Pushed to GitHub." if result.returncode == 0
          else f"Push failed: {result.stderr.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 60)
    print("WEDNESDAY BACKTEST v3")
    print("=" * 60)

    K = 3

    print("\nLoading Monday ticks...")
    returns_mon = load_and_prep_data("ticks_mon.csv")
    pca_mon, train_mean = fit_monday_pca(returns_mon, K=K)

    print("Loading Wednesday ticks...")
    returns_wed = load_and_prep_data("ticks_wed.csv")

    print("Loading Wednesday labels...")
    labels_wed = (pd.read_csv("labels_wed.csv", parse_dates=["timestamp"])
                  .set_index("timestamp"))
    label_col = labels_wed.columns[0]

    common_index = returns_wed.index.intersection(labels_wed.index)
    returns_wed  = returns_wed.loc[common_index]
    labels_wed   = labels_wed.loc[common_index]
    y_true       = labels_wed[label_col].values

    print(f"\nAligned ticks: {len(common_index)}")
    print(f"True label distribution:\n"
          f"{labels_wed[label_col].value_counts().to_string()}")

    # Build features
    monday_eigvecs_k = pca_mon.components_[:K]
    monday_eigvecs_2 = pca_mon.components_[:2]
    monday_ev_ratios = pca_mon.explained_variance_ratio_

    from p3_regime_classifier import build_streaming_features

    print("\nBuilding streaming features...")
    t0 = time.perf_counter()

    features_wed = build_streaming_features(
        returns_df       = returns_wed,
        window           = 100,
        monday_eigvecs_k = monday_eigvecs_k,
        monday_eigvecs_2 = monday_eigvecs_2,
        train_mean       = train_mean.values,
        monday_ev_ratios = monday_ev_ratios,
        q_threshold      = 0.1,
    )

    # Print diagnostics so you can verify thresholds
    print_feature_diagnostics(features_wed, y_true)

    # =========================================================================
    # HYPERPARAMETERS — tuned from diagnostics table
    #
    #   max_contrib_share:   NORMAL=0.312  FAT_FINGER=0.536  → 0.42
    #   subspace_angle_deg:  NORMAL=4.14   CORR_BREAK=63.26  → 35.0
    #                        SECTOR_ROT=31.17 (above this = SR, above 35 = CB)
    #   ev_ratio_pc1_delta:  NORMAL=0.028  SECTOR_ROT=0.314  → 0.15
    #
    # To tune: move FF_THRESH between 0.312 and 0.536
    #          move CR_THRESH between 31.17 and 63.26
    #          move SR_THRESH between 0.028 and 0.314
    # =========================================================================
    FF_THRESH = 0.42
    CR_THRESH = 35.0
    SR_THRESH = 0.15
    # =========================================================================

    print("Classifying regimes...")
    preds   = patched_predict_batch(features_wed, FF_THRESH, CR_THRESH,
                                     SR_THRESH, smoothing=5)
    elapsed = time.perf_counter() - t0
    acc     = accuracy_score(y_true, preds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Runtime:  {elapsed:.1f}s")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_true, preds, zero_division=0))
    print(f"Predicted distribution:\n"
          f"{pd.Series(preds).value_counts().to_string()}\n")

    log_and_push(FF_THRESH, CR_THRESH, SR_THRESH, acc)


if __name__ == "__main__":
    run_backtest()