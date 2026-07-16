from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

"""
Standalone script that:
  1. Loads the trained cVAE from config.PATHS["vae"].
  2. Regenerates posterior predictive and generative PC score samples.
  3. Computes per-PC MMD and full N_SCORES-D joint MMD for both sample sets.
  4. Produces a corner plot (Figure 9) and prints joint MMD^2 values for
     inline manuscript reporting.
"""

import pickle
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
from hyppo.ksample import MMD

from sklearn.decomposition import PCA
from spinodal_cvae import VariationalAutoencoder

import config


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca


# Global settings for academic-quality figures
mpl.rcParams.update(
    {
        # Figure quality
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (4, 3),  # Default size in inches (tweak per journal needs)
        # Font & text
        "font.family": "serif",  # Serif fonts look more formal
        "font.serif": ["Times New Roman", "Times", "Palatino", "Computer Modern Roman"],
        "font.size": 10,  # Matches many journal requirements
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Axes & lines
        "axes.linewidth": 0.8,  # Thinner borders
        "lines.linewidth": 1.0,  # Default line width
        "lines.markersize": 4,  # Default marker size
        # Grid & ticks
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)


def plot_corner_pc(
    ref_pc: np.ndarray,
    post_pc: np.ndarray,
    gen_pc: np.ndarray,
    n_pcs: int = 5,
    out_path: str = None,
):
    """
    Corner plot covering PCs 1..n_pcs using the corner package.

    Three datasets are overlaid by calling corner.corner() sequentially,
    passing fig= on the second and third calls to share the same axes.
    """
    COLORS = {
        "ref": "tab:blue",
        "post": "tab:orange",
        "gen": "black",
    }
    LABELS = {
        "ref": "Test Set",
        "post": "Posterior Predictive",
        "gen": r"$\beta$-cVAE Generative",
    }

    rng = np.random.default_rng(42)
    max_pts = min(5000, len(post_pc))

    def _sub(arr):
        return arr[rng.choice(len(arr), max_pts, replace=len(arr) < max_pts)]

    panel_size = 2.5
    fig_size = panel_size * n_pcs
    label_fs = 18
    tick_fs = 16
    legend_fs = 20

    pc_labels = [f"PC{i + 1}" for i in range(n_pcs)]

    ref_sub = _sub(ref_pc)[:, :n_pcs]
    post_sub = _sub(post_pc)[:, :n_pcs]
    gen_sub = _sub(gen_pc)[:, :n_pcs]

    all_data = np.concatenate([ref_sub, post_sub, gen_sub], axis=0)
    shared_range = [
        (
            all_data[:, i].min() - 0.05 * (all_data[:, i].max() - all_data[:, i].min()),
            all_data[:, i].max() + 0.05 * (all_data[:, i].max() - all_data[:, i].min()),
        )
        for i in range(n_pcs)
    ]

    kwargs_common = dict(
        bins=30,
        smooth=2.0,
        smooth1d=1.0,
        labels=pc_labels,
        plot_density=False,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.393, 0.864),  # 1 std, 2 std
        label_kwargs={"fontsize": label_fs},
        range=shared_range,
    )

    fig = corner.corner(
        ref_sub,
        color=COLORS["ref"],
        fig=plt.figure(figsize=(fig_size, fig_size)),
        **kwargs_common,
    )
    corner.corner(
        post_sub,
        color=COLORS["post"],
        fig=fig,
        **kwargs_common,
    )
    corner.corner(
        gen_sub,
        color=COLORS["gen"],
        fig=fig,
        hist_kwargs={"linestyle": "--"},
        contour_kwargs={"linestyles": ["--", "--"]},
        **kwargs_common,
    )

    # increase tick label sizes on all axes
    for ax in fig.axes:
        ax.tick_params(labelsize=tick_fs)

    # place legend in the top-right axes (row=0, col=n_pcs-1), which is empty
    axs = np.array(fig.axes).reshape(n_pcs, n_pcs)
    legend_ax = axs[1, n_pcs - 2]
    legend_ax.set_visible(True)
    legend_ax.set_axis_off()
    legend_handles = [
        mlines.Line2D([], [], color=COLORS["ref"], label=LABELS["ref"]),
        mlines.Line2D([], [], color=COLORS["post"], label=LABELS["post"]),
        mlines.Line2D([], [], color=COLORS["gen"], label=LABELS["gen"], linestyle="--"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=legend_fs,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved corner plot → {out_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    # ------------------------------------------------------------------ paths ---
    OUT_DIR = config.PATHS["figures_dir"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_SCORES = config.N_SCORES
    N_SAMPLES_PER_COND = 1000

    # --------------------------------------------------------- load artifacts ---
    with open(config.PATHS["scaler_scores"], "rb") as f:
        scaler_scores = pickle.load(f)

    with open(config.PATHS["scaler_params"], "rb") as f:
        scaler_params = pickle.load(f)

    with open(config.PATHS["vae"], "rb") as f:
        model: VariationalAutoencoder = pickle.load(f)

    print(f"Latent dim: {model.latent_dim}")

    # -------------------------------------------------------------- load data ---
    with open(config.PATHS["pca"], "rb") as f:
        pca: PCA = pickle.load(f)
        pca = trim_pca(pca, N_SCORES)

    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        scores_test = f["hetero_only"]["test"]["scores"][:, :N_SCORES]
        params_test = f["hetero_only"]["test"]["params"][...]

    print(f"scores_test: {scores_test.shape}, params_test: {params_test.shape}")

    # ------------------------------------------------- scale for model input ---
    scores_test_scaled = scaler_scores.transform(scores_test)
    params_test_scaled = scaler_params.transform(params_test)

    n_val = len(params_test_scaled)
    latent_dim = model.latent_dim
    params_expanded = np.repeat(params_test_scaled, N_SAMPLES_PER_COND, axis=0)

    # ------------------------------------------ posterior predictive samples ---
    z_mu, z_logvar = model.encode(X=scores_test_scaled, c=params_test_scaled)
    z_std = np.exp(0.5 * z_logvar)

    eps = np.random.randn(n_val, N_SAMPLES_PER_COND, latent_dim)
    z_post = (z_mu[:, None, :] + z_std[:, None, :] * eps).reshape(-1, latent_dim)

    recon_scores = scaler_scores.inverse_transform(
        model.decode(z=z_post, c=params_expanded)
    )

    # -------------------------------------------- conditional prior samples ---
    z_prior = np.random.randn(n_val * N_SAMPLES_PER_COND, latent_dim)
    gen_scores = scaler_scores.inverse_transform(
        model.decode(z=z_prior, c=params_expanded)
    )

    print(f"recon_scores: {recon_scores.shape}, gen_scores: {gen_scores.shape}")

    # ================================================================= MMD ====
    # Joint N_SCORES-D MMD via hyppo (Gretton et al. 2012, unbiased estimator, Gaussian kernel)

    rng_mmd = np.random.default_rng(0)
    max_pts = 2000

    def _subsample(arr):
        if len(arr) > max_pts:
            return arr[rng_mmd.choice(len(arr), max_pts, replace=False)]
        return arr

    print("Computing joint MMD (hyppo)…")
    stat_post, pval_post = MMD().test(
        _subsample(scores_test).astype(np.float64),
        _subsample(recon_scores).astype(np.float64),
        workers=-1,
    )
    stat_gen, pval_gen = MMD().test(
        _subsample(scores_test).astype(np.float64),
        _subsample(gen_scores).astype(np.float64),
        workers=-1,
    )
    print("MMD done.")

    print(f"\nMMD — posterior predictive vs. test : stat={stat_post:.6f}, p={pval_post:.4f}")
    print(f"MMD — generative         vs. test : stat={stat_gen:.6f},  p={pval_gen:.4f}")

    # ============================================================ PLOTS ======

    fig_corner = plot_corner_pc(
        ref_pc=scores_test,
        post_pc=recon_scores,
        gen_pc=gen_scores,
        n_pcs=10,
        out_path=OUT_DIR / "figure_9_10PC_corner.png",
    )

    fig_corner = plot_corner_pc(
        ref_pc=scores_test,
        post_pc=recon_scores,
        gen_pc=gen_scores,
        n_pcs=5,
        out_path=OUT_DIR / "figure_9_5PC_corner.png",
    )
