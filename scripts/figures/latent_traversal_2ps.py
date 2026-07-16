# Single-latent-dimension traversal -> decoded 2PS central-row cross-sections,
# for a few representative conditions. Ported to the public repo.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import config
from src_plotting.vae_utils import decode_single_dim_traversal, plot_traversal_2ps_overlay  # noqa: F401

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


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca


if __name__ == "__main__":
    N_SCORES = config.N_SCORES
    N_TRAVERSE = 100
    LATENT_RANGE = (-3, 3)
    N_CONDITIONS = 2  # number of representative conditions to traverse under

    OUT_DIR = config.FIGURES_DIR / "latent_traversal_2ps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model, scalers, pca
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

    # Load test data
    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        scores_test = f["hetero_only"]["test"]["scores"][:, :N_SCORES]
        params_test = f["hetero_only"]["test"]["params"][...]

    print(f"scores_test: {scores_test.shape}, params_test: {params_test.shape}")

    # Select representative conditions -- KMeans(n_clusters=5) + closest-to-centroid,
    # for consistency across figures.
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(params_test)
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, params_test)
    representative_indices = closest_indices[:N_CONDITIONS]

    print(f"Representative test indices: {representative_indices}")

    # Compute top-KL dims (for ranking/discussion only -- all dims are still traversed)
    scores_scaled_test = scaler_scores.transform(scores_test)
    params_scaled_test = scaler_params.transform(params_test)
    z_mu_test, z_logvar_test = model.encode(X=scores_scaled_test, c=params_scaled_test)
    kl = -0.5 * (1 + z_logvar_test - z_mu_test**2 - np.exp(z_logvar_test)).mean(axis=0)
    top_idx = np.argsort(kl)[::-1]
    print(f"Dims ranked by KL (descending, 0-indexed): {top_idx}")
    print(f"Top 5: {top_idx[:5]}, KL values: {kl[top_idx[:5]].round(4)}")

    # Traverse all dims for each representative condition
    for cond_i, test_idx in enumerate(representative_indices):
        base_score = scores_test[test_idx : test_idx + 1]
        base_param = params_test[test_idx : test_idx + 1]

        base_score_scaled = scaler_scores.transform(base_score)
        base_param_scaled = scaler_params.transform(base_param)

        base_z, _ = model.encode(X=base_score_scaled, c=base_param_scaled)
        base_z = base_z[0]  # (latent_dim,)
        base_cond_scaled = base_param_scaled[0]  # (cond_dim,)

        cond_out_dir = OUT_DIR / f"condition_{cond_i}_test_idx_{test_idx}"
        cond_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Condition {cond_i} (test_idx={test_idx}) ===")

        for dim_idx in range(latent_dim):
            traverse_range, cross_sections = decode_single_dim_traversal(
                model=model,
                score_scaler=scaler_scores,
                pca=pca,
                dim_idx=dim_idx,
                base_z=base_z,
                base_cond_scaled=base_cond_scaled,
                latent_range=LATENT_RANGE,
                n_traverse=N_TRAVERSE,
            )

            rank = int(np.where(top_idx == dim_idx)[0][0])  # 0 = highest KL
            fig_title = f"2PS Cross-Section vs. Latent Dim {dim_idx} Traversal (KL rank {rank})"

            fig, ax = plt.subplots(figsize=(6, 4.5))
            norm_ = plt.Normalize(vmin=traverse_range.min(), vmax=traverse_range.max())
            cmap_ = plt.get_cmap("plasma")
            for i, v in enumerate(traverse_range):
                ax.plot(cross_sections[i], color=cmap_(norm_(v)), linewidth=0.8, alpha=0.85)
            sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm_)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(f"$z_{{{dim_idx}}}$ value")
            ax.set_xlabel("Pixel Index (Column)")
            ax.set_ylabel("Intensity")
            ax.set_title(fig_title)
            plt.tight_layout()

            fig.savefig(cond_out_dir / f"dim_{dim_idx:02d}.png", bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {latent_dim} traversal figures -> {cond_out_dir}")

    print("\nDone. Inspect figures under:", OUT_DIR)
