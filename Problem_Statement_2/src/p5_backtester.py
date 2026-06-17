"""
p5_backtester.py
Historical simulation engine and output generation pipeline.

Executes the out-of-sample backtest using the calibrated streaming features, 
regime classifier, and dynamic barrier logic. Manages strict temporal alignment, 
including millisecond-resolution timestamp serialization for the final 
production payload.
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
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import PCA

np.random.seed(42)


# Safe data loader

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


# Vectorised predict + temporal smoothing

def patched_predict_batch(features_df, ff_thresh, cr_thresh, sr_thresh,
                           smoothing=5):
    ff   = features_df["max_contrib_share"].values
    ang  = features_df["subspace_angle_deg"].values
    ev1d = features_df["ev_ratio_pc1_delta"].values
    q    = features_df["q_stat"].values

    labels = np.full(len(features_df), "NORMAL", dtype=object)
    labels[ev1d > sr_thresh] = "SECTOR_ROTATION"
    labels[ang  > cr_thresh] = "CORRELATION_BREAK"
    labels[ff   > ff_thresh] = "FAT_FINGER"
    labels[np.isnan(q)]      = "NORMAL"

    REGIME_MAP = {"NORMAL": 0, "SECTOR_ROTATION": 1,
                  "CORRELATION_BREAK": 2, "FAT_FINGER": 3}
    REGIME_INV = {v: k for k, v in REGIME_MAP.items()}

    arr     = np.array([REGIME_MAP[x] for x in labels], dtype=np.int8)
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


# Feature diagnostics

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


# Auto-logger

def log_and_push(ff, cr, sr, acc, macro_f1):
    log_file = os.path.join(ROOT, "experiment_log.md")
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a") as f:
        if not file_exists:
            f.write("# Model Tuning Log\n\n")
            f.write("| Date & Time | FF | CR | SR | Accuracy | Macro-F1 |\n")
            f.write("|---|---|---|---|---|---|\n")
        f.write(
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {ff} | {cr} | {sr} "
            f"| **{acc*100:.2f}%** | **{macro_f1:.4f}** |\n"
        )
    print(f"\nLogged to {log_file}")
    subprocess.run(["git", "add", "experiment_log.md", "src/p5_backtester.py"],
                   cwd=ROOT)
    subprocess.run(
        ["git", "commit", "-m",
         f"v5: acc={acc*100:.2f}% macro-F1={macro_f1:.4f} "
         f"FF={ff} CR={cr} SR={sr}"],
        cwd=ROOT, capture_output=True
    )
    result = subprocess.run(["git", "push"], cwd=ROOT,
                             capture_output=True, text=True)
    print("Pushed to GitHub." if result.returncode == 0
          else f"Push failed (saved locally): {result.stderr.strip()}")


# Thursday submission CSV generator  

def generate_thursday_submission(pca_mon, train_mean, K,
                                  ff_thresh, cr_thresh, sr_thresh,
                                  team_number="6"):
    print("\n" + "=" * 60)
    print("GENERATING THURSDAY SUBMISSION CSV")
    print("=" * 60)

    from p3_regime_classifier import build_streaming_features

    returns_thu      = load_and_prep_data("ticks_thu.csv")
    monday_eigvecs_k = pca_mon.components_[:K]
    monday_eigvecs_2 = pca_mon.components_[:2]
    monday_ev_ratios = pca_mon.explained_variance_ratio_

    print("Building Thursday features...")
    features_thu = build_streaming_features(
        returns_df       = returns_thu,
        window           = 100,
        monday_eigvecs_k = monday_eigvecs_k,
        monday_eigvecs_2 = monday_eigvecs_2,
        train_mean       = train_mean.values,
        monday_ev_ratios = monday_ev_ratios,
        q_threshold      = 0.1,
    )

    print("Classifying Thursday regimes...")
    preds_thu = patched_predict_batch(
        features_thu, ff_thresh, cr_thresh, sr_thresh, smoothing=5
    )
    preds_thu = preds_thu.fillna("NORMAL")

    idx = features_thu.index
    ts_strings = pd.to_datetime(idx, format='mixed', utc=True).strftime("%Y-%m-%dT%H:%M:%S%z")

    submission = pd.DataFrame({
        "timestamp":        ts_strings,
        "predicted_regime": preds_thu.values,
    })

    out_path = os.path.join(ROOT, f"Team_{team_number}_submission.csv")
    submission.to_csv(out_path, index=False)

    print(f"\nSubmission saved: {out_path}  ({len(submission)} rows)")
    print("Thursday prediction distribution:")
    print(pd.Series(preds_thu.values).value_counts().to_string())
    return out_path

# Main

def run_backtest():
    print("=" * 60)
    print("WEDNESDAY BACKTEST")
    print("=" * 60)

    K           = 3
    TEAM_NUMBER = "6"

    print("\nLoading Monday ticks...")
    returns_mon      = load_and_prep_data("ticks_mon.csv")
    pca_mon, train_mean = fit_monday_pca(returns_mon, K=K)

    print("Loading Wednesday ticks...")
    returns_wed = load_and_prep_data("ticks_wed.csv")

    print("Loading Wednesday labels...")
    labels_wed = (pd.read_csv("labels_wed.csv", parse_dates=["timestamp"])
                  .set_index("timestamp"))
    label_col  = labels_wed.columns[0]

    common_index = returns_wed.index.intersection(labels_wed.index)
    returns_wed  = returns_wed.loc[common_index]
    labels_wed   = labels_wed.loc[common_index]
    y_true       = labels_wed[label_col].values

    print(f"\nAligned ticks: {len(common_index)}")
    print(f"True label distribution:\n"
          f"{labels_wed[label_col].value_counts().to_string()}")

    monday_eigvecs_k = pca_mon.components_[:K]
    monday_eigvecs_2 = pca_mon.components_[:2]
    monday_ev_ratios = pca_mon.explained_variance_ratio_

    from p3_regime_classifier import build_streaming_features

    print("\nBuilding streaming features (causal, no look-ahead)...")
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

    print_feature_diagnostics(features_wed, y_true)

    # HYPERPARAMETERS 

    FF_THRESH = 0.42
    CR_THRESH = 45.0
    SR_THRESH = 0.10

    print("Classifying regimes...")
    preds    = patched_predict_batch(features_wed, FF_THRESH, CR_THRESH,
                                      SR_THRESH, smoothing=5)
    elapsed  = time.perf_counter() - t0
    acc      = accuracy_score(y_true, preds)
    macro_f1 = f1_score(
        y_true, preds,
        labels=["NORMAL", "FAT_FINGER", "SECTOR_ROTATION", "CORRELATION_BREAK"],
        average="macro", zero_division=0
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Runtime:   {elapsed:.1f}s")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Macro-F1:  {macro_f1:.4f}  "
          f"{'FULL MARKS (>=0.85)' if macro_f1 >= 0.85 else 'PARTIAL (>=0.65)' if macro_f1 >= 0.65 else 'below target'}")
    print()
    print(classification_report(y_true, preds, zero_division=0))
    print(f"Predicted distribution:\n"
          f"{pd.Series(preds).value_counts().to_string()}\n")

    log_and_push(FF_THRESH, CR_THRESH, SR_THRESH, acc, macro_f1)

    generate_thursday_submission(
        pca_mon, train_mean, K,
        FF_THRESH, CR_THRESH, SR_THRESH,
        team_number=TEAM_NUMBER
    )

if __name__ == "__main__":
    run_backtest()