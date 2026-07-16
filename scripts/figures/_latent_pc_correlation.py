# Latent-dimension vs. PC-score Pearson correlation heatmap (+ per-dim KL bar,
# + line plot). Ported to the public repo; loads the trained artifacts from
# config.PATHS and writes figures into config.FIGURES_DIR.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config

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
    N_SCORES = config.N_SCORES
    n_pcs = N_SCORES

    OUT_DIR = config.FIGURES_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and scalers
    with open(config.PATHS["vae"], "rb") as f:
        model = pickle.load(f)
    with open(config.PATHS["scaler_scores"], "rb") as f:
        scaler_scores = pickle.load(f)
    with open(config.PATHS["scaler_params"], "rb") as f:
        scaler_params = pickle.load(f)

    print("Model and scalers loaded.")

    # Load test data
    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        scores_test = f["hetero_only"]["test"]["scores"][:, :N_SCORES]
        params_test = f["hetero_only"]["test"]["params"][...]

    print(f"scores_test: {scores_test.shape}, params_test: {params_test.shape}")

    # Encode test set
    scores_scaled = scaler_scores.transform(scores_test)
    params_scaled = scaler_params.transform(params_test)
    z_mu, z_logvar = model.encode(X=scores_scaled, c=params_scaled)

    print(f"z_mu: {z_mu.shape}, z_logvar: {z_logvar.shape}")

    # KL divergence per latent dimension
    kl = -0.5 * (1 + z_logvar - z_mu**2 - np.exp(z_logvar)).mean(axis=0)

    top_idx = np.argsort(kl)[::-1][:5]
    print(f"Top 5 KL dims: indices {top_idx}, values {kl[top_idx].round(4)}")

    # Compute Pearson correlation matrix  (latent_dim x n_pcs)
    n_pcs = min(n_pcs, N_SCORES)
    scores_subset = scores_test[:, :n_pcs]

    combined = np.concatenate([z_mu.T, scores_subset.T], axis=0)
    full_corr = np.corrcoef(combined)
    latent_dim = z_mu.shape[1]
    corr = full_corr[:latent_dim, latent_dim:]  # (latent_dim, n_pcs)

    print(f"Correlation matrix shape: {corr.shape}")

    # Save correlation matrix
    npy_path = OUT_DIR / "latent_pc_correlation.npy"
    np.save(npy_path, corr)
    print(f"Saved -> {npy_path}")

    # Plot heatmap
    fig, (ax_hm, ax_kl) = plt.subplots(
        1, 2,
        figsize=(max(8, n_pcs * 0.18 + 2), latent_dim * 0.28 + 1.5),
        gridspec_kw={"width_ratios": [n_pcs, 1.5]},
    )

    sns.heatmap(
        corr,
        ax=ax_hm,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=[f"PC{i + 1}" for i in range(n_pcs)],
        yticklabels=[f"z{i}" for i in range(latent_dim)],
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        linewidths=0,
    )
    ax_hm.set_xlabel("PC score")
    ax_hm.set_ylabel("Latent dimension")
    ax_hm.set_title("Pearson Correlation: Latent Dims vs. PC Scores")
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=90, fontsize=6)
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=7)

    ax_kl.barh(np.arange(latent_dim), kl, color="steelblue", edgecolor="none", height=0.7)
    ax_kl.invert_yaxis()
    ax_kl.set_yticks([])
    ax_kl.set_xlabel("KL div.", fontsize=8)
    ax_kl.set_title("KL", fontsize=9)
    ax_kl.spines[["top", "right", "left"]].set_visible(False)
    ax_kl.tick_params(axis="x", labelsize=7)

    plt.tight_layout()

    png_path = OUT_DIR / "latent_pc_correlation.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png_path}")

    # Line plot -- Pearson r vs latent dim, one line per PC (first 5)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))

    x = np.arange(1, latent_dim + 1)

    for i in range(5):
        ax2.plot(x, corr[:, i], marker="o", markersize=2, label=f"PC{i + 1}")

    # +/-1 std band computed globally across all latent dims and all PCs
    global_std = corr.std()
    ax2.axhspan(-global_std, global_std, color="gray", alpha=0.2, label=r"$\pm 1\sigma$", zorder=-10)

    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("Latent Dimension")
    ax2.set_ylabel("Pearson Correlation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.legend(frameon=False, ncol=2)

    plt.tight_layout()

    png_path2 = OUT_DIR / "latent_pc_correlation_lines.png"
    fig2.savefig(png_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {png_path2}")
