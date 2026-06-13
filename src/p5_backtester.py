"""
p5_backtester.py
================
Wednesday backtest harness with auto-logging and GitHub push.

Usage:
    python src/p5_backtester.py

Tune the four thresholds in the HYPERPARAMETERS block and re-run.
Every run is logged to experiment_log.md and pushed to GitHub.
"""

import sys
import os

# Make src/ importable when run from project root OR from inside src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Change working directory to project root so CSV paths resolve correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import pandas as pd
import numpy as np
import time
import subprocess
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score

# ── Internal imports — handle both naming conventions ────────────────────────
from p1_blind_cartographer import load_and_pivot, fit_pca_on_monday

# Aliases so the rest of the file reads cleanly
def load_and_prep_data(path):
    return load_and_pivot(path)

def fit_monday_pca(returns_mon):
    pca = fit_pca_on_monday(returns_mon)
    train_mean = pd.Series(
        returns_mon.values.mean(axis=0),
        index=returns_mon.columns
    )
    return pca, train_mean

from p3_regime_classifier import build_streaming_features, RegimeClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Auto-logger — writes to experiment_log.md and pushes to GitHub
# ─────────────────────────────────────────────────────────────────────────────

def log_and_push_to_github(ff, cr, sr, q, acc, report_str):
    log_file = os.path.join(ROOT, "experiment_log.md")
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a") as f:
        if not file_exists:
            f.write("# Model Tuning Log\n\n")
            f.write("| Date & Time | FF Thresh | CR Thresh | SR Thresh |"
                    " Q Thresh | Overall Accuracy |\n")
            f.write("|---|---|---|---|---|---|\n")
        f.write(
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {ff} | {cr} | {sr} | {q} | **{acc*100:.2f}%** |\n"
        )

    print(f"\nLogged results to {log_file}")

    print("Pushing to GitHub...")
    subprocess.run(["git", "add", "experiment_log.md", "src/p5_backtester.py"],
                   cwd=ROOT)
    commit_msg = f"Tuning run: Accuracy {acc*100:.2f}% | FF={ff} CR={cr} SR={sr} Q={q}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=ROOT)
    result = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        print("Successfully pushed to GitHub.")
    else:
        print(f"Git push failed: {result.stderr.strip()}")
        print("Results are still saved locally in experiment_log.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 60)
    print("WEDNESDAY BACKTEST")
    print("=" * 60)

    # ── Load data — CSVs are in project ROOT ─────────────────────────────────
    # The script auto-chdirs to ROOT above so these paths always work
    # regardless of whether you run from root or from src/
    print("\nLoading Monday ticks...")
    returns_mon = load_and_prep_data("ticks_mon.csv")
    pca_mon, train_mean = fit_monday_pca(returns_mon)

    print("Loading Wednesday ticks...")
    returns_wed = load_and_prep_data("ticks_wed.csv")

    print("Loading Wednesday labels...")
    labels_wed = (pd.read_csv("labels_wed.csv", parse_dates=["timestamp"])
                  .set_index("timestamp"))
    label_col = labels_wed.columns[0]

    # Align timestamps
    common_index = returns_wed.index.intersection(labels_wed.index)
    returns_wed  = returns_wed.loc[common_index]
    labels_wed   = labels_wed.loc[common_index]
    y_true       = labels_wed[label_col].values

    print(f"\nAligned ticks: {len(common_index)}")
    print(f"True label distribution:\n{labels_wed[label_col].value_counts().to_string()}")

    # ── Build features ────────────────────────────────────────────────────────
    K = 3   # must match what you chose in Part 1
    monday_eigvecs_k = pca_mon.components_[:K]
    monday_eigvecs_2 = pca_mon.components_[:2]
    monday_ev_ratios = pca_mon.explained_variance_ratio_

    print("\nBuilding streaming features (causal, no look-ahead)...")
    t0 = time.perf_counter()

    features_wed = build_streaming_features(
        returns_df       = returns_wed,
        window           = 100,
        monday_eigvecs_k = monday_eigvecs_k,
        monday_eigvecs_2 = monday_eigvecs_2,
        train_mean       = train_mean.values,
        monday_ev_ratios = monday_ev_ratios,
        q_threshold      = 0.1,      # used only for info, not filtering
    )

    # =========================================================================
    # HYPERPARAMETERS — tune these numbers and re-run
    #
    # What each one does:
    #   FF_THRESH  — max_contrib_share threshold for FAT_FINGER detection.
    #                Lower = more sensitive to single-asset spikes.
    #                Too low = false positives on normal noise.
    #                Start: 0.35. Try: 0.20, 0.15, 0.10
    #
    #   CR_THRESH  — subspace_angle_deg threshold for CORRELATION_BREAK.
    #                Lower = fires earlier on eigenvector rotation.
    #                Start: 20.0. Try: 15.0, 25.0, 30.0
    #
    #   SR_THRESH  — ev_ratio_pc1_delta threshold for SECTOR_ROTATION.
    #                Lower = more sensitive to eigenvalue spectrum shifts.
    #                Start: 0.05. Try: 0.03, 0.08, 0.10
    #
    #   Q_THRESH   — minimum Q-stat required to confirm FAT_FINGER.
    #                Lower = easier to trigger FAT_FINGER.
    #                Start: 0.001. Try: 0.0005, 0.002
    # =========================================================================
    FF_THRESH = 0.05
    CR_THRESH = 45.0
    SR_THRESH = 0.12
    Q_THRESH  = 0.001
    # =========================================================================

    print("Classifying regimes...")
    classifier = RegimeClassifier(
        thresholds={
            "ff_thresh": FF_THRESH,
            "cr_thresh": CR_THRESH,
            "sr_thresh": SR_THRESH,
        },
        q_thresh  = Q_THRESH,
        smoothing = 5,
    )
    preds = classifier.predict_batch(features_wed)

    elapsed = time.perf_counter() - t0
    acc     = accuracy_score(y_true, preds)
    report  = classification_report(y_true, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Runtime:          {elapsed:.1f}s")
    print(f"Overall accuracy: {acc*100:.2f}%\n")
    print(report)

    print(f"\nPredicted label distribution:\n{pd.Series(preds).value_counts().to_string()}")

    # ── Log and push ──────────────────────────────────────────────────────────
    log_and_push_to_github(FF_THRESH, CR_THRESH, SR_THRESH, Q_THRESH, acc, report)


if __name__ == "__main__":
    run_backtest()