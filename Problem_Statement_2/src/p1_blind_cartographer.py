"""
PS2-Part 1: The Blind Cartographer 

This module handles the initial exploratory PCA on Monday's tick data.

Core tasks covered:
- (a) Data loading, PCA fitting, and K-selection (Kaiser / broken-stick / parallel)
- (b) Analyzing the 12-gon polygon geometry and rotation ambiguity in the (PC1, PC2) plane
- (c) Quantifying information leakage (ev_test_clean vs ev_test_leaky)
- (d) Proving the equal-weight portfolio neutrality under the 12-gon assumption
- (e) Supporting logic for notebook observations
"""
#imports used by the notebook and the bonus pipeline 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

np.random.seed(0)   # mandatory for parallel-analysis resampling

# (a) Data loading and PCA fit


def load_and_pivot(csv_path: str, reference_columns=None) -> pd.DataFrame:
    """
    -Load a ticks CSV and pivot into a (T x 12) returns matrix.
    Drops any timestamp where any  asset has a NaN mid_return.
    If reference_columns is given, reindex to that column order
    -(Critical: Tuesday/Wed/Thu must match Monday's asset order for
    the eigenvectors fitted on Monday to align correctly).
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    pivoted = (df
               .pivot(index="timestamp", columns="asset_id", values="mid_return")
               .dropna())
    if reference_columns is not None:
        pivoted = pivoted.reindex(columns=reference_columns)
    return pivoted


def fit_pca_on_monday(returns_mon: pd.DataFrame) -> PCA:
    """
    Fit PCA on centred (NOT standardised) Monday returns.
    The assignment says: 'Fit PCA on the centred returns (no scaling)
    using the covariance matrix; do not standardise the columns.'
    sklearn PCA with default settings does exactly this.
    """
    pca = PCA()
    pca.fit(returns_mon.values)
    return pca

#(a) Component-count selection criteria


def select_n_components(eigenvalues: np.ndarray,
                        returns_matrix: np.ndarray,
                        n_resamples: int = 500,
                        seed: int = 0) -> dict:
    """
Returns a dictionary containing the recommended number of components (K) 
based on three standard selection criteria:

1. Kaiser Criterion (Covariance-adapted): 
   Retains components where the eigenvalue > mean eigenvalue. 
   (Note: Standard Kaiser uses > 1 for correlation PCA, but the mean 
   eigenvalue is the correct threshold for covariance PCA).

2. Broken-Stick: 
   Retains component j if λ_j > λ̄ * Σ_{i=j}^{12} (1/i), 
   where λ̄ is the mean eigenvalue.

3. Parallel Analysis (Horn's Method): 
   Runs 500 resamples, shuffling each column independently to destroy 
   cross-asset correlations while preserving marginals. Retains a 
   component if its eigenvalue > the 95th percentile of the noise.
"""
    rng = np.random.default_rng(seed)
    n, p = returns_matrix.shape
    mean_ev = eigenvalues.mean()

    #  Kaiser 
    kaiser_k = int(np.sum(eigenvalues > mean_ev))

    # Broken-stick 
    bs_thresholds = mean_ev * np.array([
        sum(1.0 / i for i in range(j, p + 1))
        for j in range(1, p + 1)
    ])
    broken_stick_k = int(np.sum(eigenvalues > bs_thresholds))

    # Parallel analysis 
    resample_evs = np.zeros((n_resamples, p))
    centred = returns_matrix - returns_matrix.mean(axis=0)
    for r in range(n_resamples):
        shuffled = np.column_stack([
            rng.permutation(centred[:, col]) for col in range(p)
        ])
        cov_r = np.cov(shuffled, rowvar=False)
        resample_evs[r] = np.sort(np.linalg.eigvalsh(cov_r))[::-1]

    pa_thresholds = np.percentile(resample_evs, 95, axis=0)
    parallel_k = int(np.sum(eigenvalues > pa_thresholds))

    return {
        "kaiser":        kaiser_k,
        "broken_stick":  broken_stick_k,
        "parallel":      parallel_k,
    }


def plot_scree(eigenvalues: np.ndarray, verdicts: dict, title="Scree plot"):
    #Scree plot annotated with each criterion's cutoff
    fig, ax = plt.subplots(figsize=(8, 4))
    k = len(eigenvalues)
    xs = np.arange(1, k + 1)
    evr = eigenvalues / eigenvalues.sum()

    ax.bar(xs, evr, color="#7F77DD", alpha=0.7, label="Individual EV ratio")
    ax.plot(xs, np.cumsum(evr), "o-", color="#D85A30", label="Cumulative EV ratio")

    colors = {"kaiser": "#1D9E75", "broken_stick": "#BA7517", "parallel": "#E24B4A"}
    for name, k_val in verdicts.items():
        ax.axvline(k_val + 0.5, color=colors[name], linestyle="--",
                   linewidth=1.2, label=f"{name}: K={k_val}")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# (b) Loadings plot + rotation ambiguity

def plot_loadings_scatter(pca: PCA, asset_ids, title="PC1 vs PC2 loadings"):
    """Plots the asset loadings in the (PC1, PC2) plane. 
    If the underlying math holds, the assets should perfectly map to the vertices 
    of a regular 12-gon, representing the 12th roots of unity.
    """
    loadings = pca.components_[:2].T    # (12, 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(loadings[:, 0], loadings[:, 1], zorder=3)
    for i, aid in enumerate(asset_ids):
        ax.annotate(aid, (loadings[i, 0], loadings[i, 1]),
                    fontsize=8, textcoords="offset points", xytext=(4, 4))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    return loadings


def demonstrate_rotation_ambiguity(pca: PCA, returns_mon: np.ndarray,
                                   angle_deg: float = 30.0):
    """(b) Rotate the top-2 loadings by `angle_deg` and show that:
    - the per-axis variance decomposition changes (PC1 ≠ PC2 anymore)
    - the joint  variance (λ1 + λ2) is invariant
    This is the rotation-ambiguity problem of unsupervised factor models.
    """
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    loadings_orig = pca.components_[:2].T   # (12, 2)
    loadings_rot  = loadings_orig @ R.T

    centred = returns_mon - returns_mon.mean(axis=0)
    scores_orig = centred @ loadings_orig   # (T, 2)
    scores_rot  = centred @ loadings_rot

    var_orig = scores_orig.var(axis=0)
    var_rot  = scores_rot.var(axis=0)

    print(f"Original  — var(PC1)={var_orig[0]:.6f}  var(PC2)={var_orig[1]:.6f}"
          f"  sum={var_orig.sum():.6f}")
    print(f"Rotated   — var(ax1)={var_rot[0]:.6f}  var(ax2)={var_rot[1]:.6f}"
          f"  sum={var_rot.sum():.6f}")
    print("Joint variance is invariant under rotation ✓"
          if abs(var_orig.sum() - var_rot.sum()) < 1e-10
          else "WARNING: joint variance changed — check implementation")

    # Side-by-side scatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    asset_ids = [f"S{i:02d}" for i in range(1, 13)]
    for ax, ld, ttl in zip(axes,
                           [loadings_orig, loadings_rot],
                           ["Original loadings", f"Rotated {angle_deg}°"]):
        ax.scatter(ld[:, 0], ld[:, 1])
        for i, aid in enumerate(asset_ids):
            ax.annotate(aid, (ld[i, 0], ld[i, 1]), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(ttl)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# (c) Information leakage quantification

def projected_explained_variance(test_X: np.ndarray,
                                  eigenvectors_k: np.ndarray,
                                  train_mean: np.ndarray) -> float:
    """
    EV ratio on test_X using only training-set statistics.

    ev = 1 - ||X_test - X̂_test||²_F / ||X_test||²_F
    where X_test is centred by train_mean (NOT test mean — that would
    be its own form of leakage), and X̂_test is its reconstruction
    from the top-K training eigenvectors.

    eigenvectors_k: (K, n_assets) — rows are eigenvectors.
    """
    X_c = test_X - train_mean
    P   = eigenvectors_k                   # (K, n)
    X_hat = X_c @ P.T @ P                  # projection back to n-dim space
    residual_var = np.sum((X_c - X_hat) ** 2)
    total_var    = np.sum(X_c ** 2)
    return float(1.0 - residual_var / total_var)


def quantify_leakage(returns_mon: pd.DataFrame,
                     returns_tue: pd.DataFrame,
                     K: int) -> dict:
    """Correct procedure: PCA on Monday only, centre Tuesday by Monday mean.
    Leaky procedure:   PCA on Mon+Tue combined, centre by combined mean.
    Returns gap = ev_test_leaky - ev_test_clean.
    """
    mon_vals = returns_mon.values
    tue_vals = returns_tue.values

    #Correct 
    pca_mon   = PCA().fit(mon_vals)
    train_mean = mon_vals.mean(axis=0)
    ev_clean  = projected_explained_variance(
        tue_vals, pca_mon.components_[:K], train_mean
    )

    # Leaky 
    combined     = np.vstack([mon_vals, tue_vals])
    pca_combined = PCA().fit(combined)
    combined_mean = combined.mean(axis=0)
    ev_leaky = projected_explained_variance(
        tue_vals, pca_combined.components_[:K], combined_mean
    )

    gap = ev_leaky - ev_clean
    print(f"ev_test_clean  = {ev_clean:.6f}")
    print(f"ev_test_leaky  = {ev_leaky:.6f}")
    print(f"Leakage gap    = {gap:.6f}")
    return {"ev_clean": ev_clean, "ev_leaky": ev_leaky, "gap": gap}



# (d) Structural neutrality proof (empirical verification)
def verify_neutrality(pca: PCA, n_assets: int = 12):
    """
    Computes the equal-weight portfolio's loading on PC1 and PC2.
    Under the population 12-gon assumption these should be ~0.
    Algebraic proof (for the markdown cell):
        If the 12 assets sit at e^{i·2πk/12} for k=0…11 in the (PC1,PC2) plane,
        then the equal-weight loading is:
            w^T v1 = (1/12) Σ cos(2πk/12) = Re[(1/12) Σ e^{i·2πk/12}] = 0
            w^T v2 = (1/12) Σ sin(2πk/12) = Im[(1/12) Σ e^{i·2πk/12}] = 0
        because Σ_{k=0}^{11} e^{i·2πk/12} = 0 (sum of all 12th roots of unity).
    """
    loadings = pca.components_[:2].T   # (12, 2)
    w_eq = np.ones(n_assets) / n_assets
    beta_pc1 = float(w_eq @ loadings[:, 0])
    beta_pc2 = float(w_eq @ loadings[:, 1])
    print(f"Equal-weight loading on PC1: {beta_pc1:.3e}  (expected ≈ 0)")
    print(f"Equal-weight loading on PC2: {beta_pc2:.3e}  (expected ≈ 0)")
    return beta_pc1, beta_pc2