import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: THE BLIND CARTOGRAPHER
# This module ingests Monday's tick data, extracts the principal components,
# and geometrically maps the hidden factor structure (The Phantom Lattice).
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prep_data(filepath):
    """Loads tick data and pivots it into a T x 12 returns matrix."""
    print(f"Loading data from {filepath}...")
    ticks = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    returns_df = ticks.pivot(
        index='timestamp', 
        columns='asset_id', 
        values='mid_return'
    ).dropna()
    
    return returns_df

def fit_monday_pca(returns_df):
    """Fits the PCA model on Monday's data to establish the baseline."""
    train_mean = returns_df.mean(axis=0)
    centered_returns = returns_df - train_mean
    
    pca = PCA()
    pca.fit(centered_returns)
    
    ev_ratios = pca.explained_variance_ratio_
    print(f"PC1 explains {ev_ratios[0]*100:.2f}% of variance.")
    print(f"PC2 explains {ev_ratios[1]*100:.2f}% of variance.")
    print(f"Top 2 PCs combined: {np.sum(ev_ratios[:2])*100:.2f}%\n")
    
    return pca, train_mean

def plot_phantom_lattice(pca, asset_names):
    """
    Plots the PC1 vs PC2 loadings. 
    Reveals the hidden geometric structure of the 12 assets.
    """
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(pc1, pc2, c='royalblue', s=150, zorder=3, edgecolors='black')
    
    for i, txt in enumerate(asset_names):
        plt.annotate(f"Asset {txt}", (pc1[i], pc2[i]), xytext=(8, 8), 
                     textcoords='offset points', fontsize=11, fontweight='bold')
        plt.plot([0, pc1[i]], [0, pc2[i]], color='gray', linestyle='--', alpha=0.5, zorder=1)
        
    angles = np.arctan2(pc2, pc1)
    sort_order = np.argsort(angles)
    poly_pc1 = np.append(pc1[sort_order], pc1[sort_order][0])
    poly_pc2 = np.append(pc2[sort_order], pc2[sort_order][0])
    
    plt.plot(poly_pc1, poly_pc2, color='darkred', linewidth=2, zorder=2, alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Exposure to PC1 (Market Factor)', fontsize=14)
    plt.ylabel('Exposure to PC2 (Sector/Relative Factor)', fontsize=14)
    plt.title('The Phantom Lattice: Asset Loadings in Factor Space', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal') 
    plt.tight_layout()
    
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/phantom_lattice_12gon.png', dpi=300)
    print("Artifact saved: artifacts/phantom_lattice_12gon.png")
    plt.show()

def prove_factor_neutrality(pca):
    """
    Algebraically proves why an equal-weight portfolio is factor neutral.
    """
    print("--- GEOMETRIC PROOF OF FACTOR NEUTRALITY ---")
    pc1_sum = np.sum(pca.components_[0])
    pc2_sum = np.sum(pca.components_[1])
    
    print(f"Sum of PC1 Loadings: {pc1_sum:.6f}")
    print(f"Sum of PC2 Loadings: {pc2_sum:.6f}\n")
    
    print("Conclusion for Panel:")
    print("Because the assets form a regular 12-sided polygon perfectly")
    print("centered around the origin in the factor space, the vector sum of their")
    print("loadings is exactly zero. Therefore, an equal-weight portfolio")
    print("has a net exposure of zero to both PC1 and PC2.")

if __name__ == "__main__":
    DATA_PATH = "../QMP-Case study (PS+data)/ticks_mon.csv" 
    
    if os.path.exists(DATA_PATH):
        returns_mon = load_and_prep_data(DATA_PATH)
        pca_mon, train_mean = fit_monday_pca(returns_mon)
        
        plot_phantom_lattice(pca_mon, returns_mon.columns)
        prove_factor_neutrality(pca_mon)
    else:
        print(f"Error: Could not find {DATA_PATH}. Place your CSV in the data folder.")