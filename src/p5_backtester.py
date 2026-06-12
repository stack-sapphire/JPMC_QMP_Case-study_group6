import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import time
import os
import subprocess
from datetime import datetime

# Import your custom modules
from p1_blind_cartographer import load_and_prep_data, fit_monday_pca
from p3_regime_classifier import build_streaming_features, RegimeClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: THE REAL-DATA BACKTESTER & AUTO-LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def log_and_push_to_github(ff, cr, sr, q, acc):
    """Automatically logs the run to a Markdown file and pushes to GitHub."""
    # This specifically forces the file to stay in your project's main folder
    log_file = "experiment_log.md" 
    
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, "a") as f:
        if not file_exists:
            f.write("# 🧪 Model Tuning Log\n\n")
            f.write("| Date & Time | FF Thresh | CR Thresh | SR Thresh | Q Thresh | Overall Accuracy |\n")
            f.write("|---|---|---|---|---|---|\n")
        f.write(f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {ff} | {cr} | {sr} | {q} | **{acc*100:.2f}%** |\n")
        
    print(f"\n📝 Logged results locally to {log_file}")
    
    print("☁️ Pushing iteration to GitHub...")
    subprocess.run(["git", "add", "experiment_log.md", "src/p5_backtester.py"])
    commit_msg = f"Tuning iteration: Accuracy {acc*100:.2f}%"
    subprocess.run(["git", "commit", "-m", commit_msg])
    subprocess.run(["git", "push"])
    print("✅ Successfully synced to GitHub!")
def run_backtest():
    print("="*60)
    print("🚀 INITIATING WEDNESDAY BACKTEST 🚀")
    print("="*60)
    
    DATA_DIR = "QMP-Case study (PS+data)"
    mon_path = f"{DATA_DIR}/ticks_mon.csv"
    wed_path = f"{DATA_DIR}/ticks_wed.csv"
    labels_path = f"{DATA_DIR}/labels_wed.csv"
    
    returns_mon = load_and_prep_data(mon_path)
    pca_mon, train_mean = fit_monday_pca(returns_mon)
    
    monday_eigvecs_k = pca_mon.components_[:10]  
    monday_eigvecs_2 = pca_mon.components_[:2]   
    monday_ev_ratios = pca_mon.explained_variance_ratio_
    
    returns_wed = load_and_prep_data(wed_path)
    print(f"Loading official labels from {labels_path}...")
    labels_wed = pd.read_csv(labels_path, parse_dates=['timestamp']).set_index('timestamp')
    label_col = labels_wed.columns[0] 
    
    common_index = returns_wed.index.intersection(labels_wed.index)
    returns_wed = returns_wed.loc[common_index]
    labels_wed = labels_wed.loc[common_index]
    y_true = labels_wed[label_col].values
    
    print("\nExtracting streaming features for Wednesday...")
    start_time = time.perf_counter()
    
    features_wed = build_streaming_features(
        returns_df=returns_wed, window=100, 
        monday_eigvecs_k=monday_eigvecs_k, monday_eigvecs_2=monday_eigvecs_2,
        train_mean=train_mean.values, monday_ev_ratios=monday_ev_ratios, q_threshold=0.1
    )
    
    # =========================================================================
    # 🎛️ YOUR HYPERPARAMETERS (TUNE THESE NUMBERS!)
    # =========================================================================
    FF_THRESH = 0.05   # Lowered significantly to catch Fat Fingers
    CR_THRESH = 45.0   
    SR_THRESH = 0.12   
    Q_THRESH  = 0.001  # Lowered to filter less noise during Fat Fingers
    
    print("Classifying market regimes...")
    classifier = RegimeClassifier(ff_thresh=FF_THRESH, cr_thresh=CR_THRESH, sr_thresh=SR_THRESH, q_thresh=Q_THRESH)
    preds = classifier.predict_batch(features_wed)
    # =========================================================================
    
    latency = time.perf_counter() - start_time
    acc = accuracy_score(y_true, preds)
    
    print("\n" + "="*60)
    print("🏆 FINAL BACKTEST RESULTS")
    print("="*60)
    print(f"Overall Accuracy:      {acc*100:.2f}%\n")
    print(classification_report(y_true, preds, zero_division=0))
    
    log_and_push_to_github(FF_THRESH, CR_THRESH, SR_THRESH, Q_THRESH, acc)

if __name__ == "__main__":
    run_backtest()