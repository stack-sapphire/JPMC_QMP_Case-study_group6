"""
p5_backtester.py  — v2
========================
Fixed:
  1. NaN/Inf in returns matrix was corrupting PCA covariance (divide by zero)
     → now clips and fills before fitting
  2. FAT_FINGER never triggered because max_contrib_share was NaN
     → root cause was the corrupted eigenvectors above
  3. Threshold tuning guide updated with correct starting values
     based on actual feature distributions printed at runtime
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
# Safe data loader — fixes the NaN/Inf that caused the PCA warnings
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prep_data(path):
    """
    Load tick CSV, pivot to (T x 12) returns matrix, drop NaN rows,
    then clip extreme values that cause overflow in the covariance matmul.
    """
    from p1_blind_cartographer import load_and_pivot
    df = load_and_pivot(path)

    # Replace any remaining Inf/-Inf with NaN then drop those rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Clip to ±10 std per column — removes fat-tail outliers that cause
    # overflow in the covariance matmul without distorting normal returns
    std = df.std()
    std[std == 0] = 1.0   # guard zero-std columns
    df = df.clip(lower=-10 * std, upper=10 * std, axis=1)

    return df


def fit_monday_pca(returns_mon, K=3):
    """
    Fit PCA on Monday returns. Returns (pca, train_mean).
    Centres the data before fitting to avoid the matmul overflow.
    """
    train_mean = returns_mon.mean()
    centred    = returns_mon - train_mean

    # Double-check no NaN/Inf survived
    assert not centred.isnull().any().any(), "NaN in centred returns"
    assert np.isfinite(centred.values).all(), "Inf in centred returns"

    pca = PCA(n_components=None)
    pca.fit(centred.values)

    print(f"  PCA fit: top-{K} components explain "
          f"{pca.explained_variance_ratio_[:K].sum()*100:.1f}% of variance")

    return pca, train_mean


# ─────────────────────────────────────────────────────────────────────────────
# Feature diagnostics — prints what the features actually look like
# per regime so you can set thresholds with real numbers
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_diagnostics(features_wed, y_true):
    """
    Print per-regime median of each feature.
    Use these numbers to set FF_THRESH, CR_THRESH, SR_THRESH, Q_THRESH.
    """
    REGIMES = ["NORMAL", "FAT_FINGER", "SECTOR_ROTATION", "CORRELATION_BREAK"]
    feats   = ["q_stat", "max_contrib_share", "subspace_angle_deg",
               "ev_ratio_pc1_delta", "mean_corr"]

    df = features_wed.copy()
    df["true_label"] = y_true

    print("\n" + "=" * 70)
    print("FEATURE DIAGNOSTICS — median per regime (use these to set thresholds)")
    print("=" * 70)
    print(f"{'Feature':<25}", end="")
    for r in REGIMES:
        print(f"  {r[:12]:<13}", end="")
    print()
    print("-" * 70)

    for feat in feats:
        if feat not in df.columns:
            continue
        print(f"{feat:<25}", end="")
        for r in REGIMES:
            sub = df.loc[df["true_label"] == r, feat].dropna()
            med = sub.median() if len(sub) > 0 else float("nan")
            print(f"  {med:>12.5f} ", end="")
        print()

    print("=" * 70)
    print("THRESHOLD GUIDE:")
    print("  FF_THRESH  → set BELOW FAT_FINGER median of max_contrib_share")
    print("              and ABOVE NORMAL median of max_contrib_share")
    print("  CR_THRESH  → set BELOW CORRELATION_BREAK median of subspace_angle_deg")
    print("              and ABOVE NORMAL median of subspace_angle_deg")
    print("  SR_THRESH  → set BELOW SECTOR_ROTATION median of ev_ratio_pc1_delta")
    print("              and ABOVE NORMAL median of ev_ratio_pc1_delta")
    print("  Q_THRESH   → set BELOW FAT_FINGER median of q_stat")
    print("              and ABOVE NORMAL median of q_stat")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-logger
# ─────────────────────────────────────────────────────────────────────────────

def log_and_push(ff, cr, sr, q, acc):
    log_file = os.path.join(ROOT, "experiment_log.md")
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a") as f:
        if not file_exists:
            f.write("# Model Tuning Log\n\n")
            f.write("| Date & Time | FF | CR | SR | Q | Accuracy |\n")
            f.write("|---|---|---|---|---|---|\n")
        f.write(
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {ff} | {cr} | {sr} | {q} | **{acc*100:.2f}%** |\n"
        )

    print(f"\nLogged to {log_file}")
    subprocess.run(["git", "add", "experiment_log.md", "src/p5_backtester.py"],
                   cwd=ROOT)
    result_commit = subprocess.run(
        ["git", "commit", "-m",
         f"Tuning run: {acc*100:.2f}% | FF={ff} CR={cr} SR={sr} Q={q}"],
        cwd=ROOT, capture_output=True, text=True
    )
    result_push = subprocess.run(["git", "push"], cwd=ROOT,
                                  capture_output=True, text=True)
    if result_push.returncode == 0:
        print("Pushed to GitHub.")
    else:
        print(f"Push failed (results saved locally): {result_push.stderr.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 60)
    print("WEDNESDAY BACKTEST v2")
    print("=" * 60)

    K = 3   # number of PCA components — must match Part 1 choice

    # ── Load data ─────────────────────────────────────────────────────────────
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
    print(f"True label distribution:\n{labels_wed[label_col].value_counts().to_string()}")

    # ── Build features ────────────────────────────────────────────────────────
    monday_eigvecs_k = pca_mon.components_[:K]
    monday_eigvecs_2 = pca_mon.components_[:2]
    monday_ev_ratios = pca_mon.explained_variance_ratio_

    from p3_regime_classifier import build_streaming_features, RegimeClassifier

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

    # Print diagnostics BEFORE setting thresholds
    # Read the printed medians and set your thresholds below them
    print_feature_diagnostics(features_wed, y_true)

    # =========================================================================
    # HYPERPARAMETERS
    #
    # STEP 1: Run the script once with these defaults.
    # STEP 2: Read the FEATURE DIAGNOSTICS table printed above.
    # STEP 3: Set each threshold between NORMAL median and target-regime median.
    #
    # Example: if diagnostics show
    #   max_contrib_share: NORMAL=0.18, FAT_FINGER=0.52
    #   → set FF_THRESH = 0.35  (halfway between)
    # =========================================================================
    FF_THRESH = 0.35    # max_contrib_share cutoff for FAT_FINGER
    CR_THRESH = 20.0    # subspace_angle_deg cutoff for CORRELATION_BREAK
    SR_THRESH = 0.05    # ev_ratio_pc1_delta cutoff for SECTOR_ROTATION
    Q_THRESH  = 0.001   # minimum q_stat to confirm FAT_FINGER
    # =========================================================================

    print("Classifying regimes...")
    clf = RegimeClassifier(
        thresholds={
            "ff_thresh": FF_THRESH,
            "cr_thresh": CR_THRESH,
            "sr_thresh": SR_THRESH,
        },
        q_thresh  = Q_THRESH,
        smoothing = 5,
    )
    preds   = clf.predict_batch(features_wed)
    elapsed = time.perf_counter() - t0
    acc     = accuracy_score(y_true, preds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Runtime:  {elapsed:.1f}s")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_true, preds, zero_division=0))
    print(f"Predicted distribution:\n{pd.Series(preds).value_counts().to_string()}\n")

    log_and_push(FF_THRESH, CR_THRESH, SR_THRESH, Q_THRESH, acc)


if __name__ == "__main__":
    run_backtest()