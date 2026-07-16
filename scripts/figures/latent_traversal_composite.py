from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# %% Imports & config
import pickle

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from src_plotting.vae_utils import decode_single_dim_traversal
import config


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Palatino", "Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    }
)


if __name__ == "__main__":
    # %% Paths & parameters
    OUT_DIR = config.PATHS["figures_dir"] / "latent_traversal_2ps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_SCORES = config.N_SCORES
    N_TRAVERSE = 100
    LATENT_RANGE = (-3, 3)
    N_CONDITIONS = 5  # the five representative conditions A-E, matching Case Study II

    # Dims to show in the composite (0-indexed array positions).
    # Display labels use the paper's 1-indexed convention (dim_idx + 1).
    #   dim 1  -> z_2  (high KL)
    #   dim 11 -> z_12 (high KL)
    #   dim 5  -> z_6  (low KL representative)
    COMPOSITE_DIMS = [1, 11, 5]
    PANEL_TAGS = ["(a)", "(b)", "(c)"]

    # %% Load model, scalers, pca
    with open(config.PATHS["vae"], "rb") as f:
        model = pickle.load(f)
    with open(config.PATHS["scaler_scores"], "rb") as f:
        scaler_scores = pickle.load(f)
    with open(config.PATHS["scaler_params"], "rb") as f:
        scaler_params = pickle.load(f)
    with open(config.PATHS["pca"], "rb") as f:
        pca = pickle.load(f)
        pca = trim_pca(pca, N_SCORES)

    latent_dim = model.latent_dim
    print(f"latent_dim: {latent_dim}")

    # %% Load test data
    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        scores_test = f["hetero_only"]["test"]["scores"][:, :N_SCORES]
        params_test = f["hetero_only"]["test"]["params"][...]

    print(f"scores_test: {scores_test.shape}, params_test: {params_test.shape}")

    # %% Select representative conditions — identical KMeans selection to Case Study II
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(params_test)
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, params_test)
    representative_indices = closest_indices[:N_CONDITIONS]
    CONDITION_LABELS = ["A", "B", "C", "D", "E"][:N_CONDITIONS]

    print(f"Representative test indices (A-E): {representative_indices}")

    # %% Compute per-dim KL and ranks (for label sanity-checking)
    scores_scaled_test = scaler_scores.transform(scores_test)
    params_scaled_test = scaler_params.transform(params_test)
    z_mu_test, z_logvar_test = model.encode(X=scores_scaled_test, c=params_scaled_test)
    kl = -0.5 * (1 + z_logvar_test - z_mu_test**2 - np.exp(z_logvar_test)).mean(axis=0)
    top_idx = np.argsort(kl)[::-1]  # descending KL; top_idx[0] = highest KL

    print(f"Dims ranked by KL (descending, 0-indexed): {top_idx}")
    for d in COMPOSITE_DIMS:
        rank = int(np.where(top_idx == d)[0][0])  # 0 = highest KL
        print(f"  dim {d} (z_{d + 1}): KL={kl[d]:.4f}, KL rank={rank} of {latent_dim}")

    lowest_kl_dim = int(top_idx[-1])
    print(f"Lowest-KL dim (0-indexed): {lowest_kl_dim} (z_{lowest_kl_dim + 1}), KL={kl[lowest_kl_dim]:.4f}")

    # %% Compute the base latent point for each representative condition A-E
    base_zs = []           # per-condition encoded latent vectors (latent_dim,)
    base_conds_scaled = []  # per-condition scaled conditioning vectors (cond_dim,)
    for cond_i, test_idx in enumerate(representative_indices):
        test_idx = int(test_idx)
        base_score_scaled = scaler_scores.transform(scores_test[test_idx : test_idx + 1])
        base_param_scaled = scaler_params.transform(params_test[test_idx : test_idx + 1])
        base_z, _ = model.encode(X=base_score_scaled, c=base_param_scaled)
        base_zs.append(base_z[0])
        base_conds_scaled.append(base_param_scaled[0])

    # %% Build composite figure: N_CONDITIONS rows (A-E) x 3 cols (z2, z12, z6).
    # Panels are short in the y-direction so the full grid stays compact. Each
    # ROW shares a y-axis so the magnitude of the induced change is directly
    # comparable across the three dimensions for a given condition -- conditions
    # have very different baseline intensities, so sharing per-column instead
    # would compress low-intensity conditions into a near-flat line.
    norm_ = plt.Normalize(vmin=LATENT_RANGE[0], vmax=LATENT_RANGE[1])
    cmap_ = plt.get_cmap("plasma")

    # ~1.35 in per row keeps the 5-row grid from becoming excessively tall.
    fig, axes = plt.subplots(
        N_CONDITIONS, 3,
        figsize=(12, 1.35 * N_CONDITIONS + 0.6),
        sharey="row",
        squeeze=False,
    )

    for col, (dim_idx, tag) in enumerate(zip(COMPOSITE_DIMS, PANEL_TAGS)):
        for row, cond_label in enumerate(CONDITION_LABELS):
            ax = axes[row, col]
            traverse_range, cross_sections = decode_single_dim_traversal(
                model=model,
                score_scaler=scaler_scores,
                pca=pca,
                dim_idx=dim_idx,
                base_z=base_zs[row],
                base_cond_scaled=base_conds_scaled[row],
                latent_range=LATENT_RANGE,
                n_traverse=N_TRAVERSE,
            )

            for i, v in enumerate(traverse_range):
                ax.plot(cross_sections[i], color=cmap_(norm_(v)), linewidth=0.7, alpha=0.85)

            # Force short (wide) panels regardless of the data's y-range.
            ax.set_box_aspect(0.32)

            # Column tag (z-dimension) only on the top row.
            if row == 0:
                ax.text(
                    0.02, 0.95, tag, transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold",
                )
            # Row label (condition A-E) on the left column.
            if col == 0:
                ax.set_ylabel(f"{cond_label}\nIntensity", fontsize=8)
            # x-label only on the bottom row.
            if row == N_CONDITIONS - 1:
                ax.set_xlabel("Pixel Index (Column)")

    # Single shared colorbar spanning all panels.
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm_)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Latent value ($z_i$)")

    out_path = OUT_DIR / "composite_traversal_A_E.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved composite figure -> {out_path}")
