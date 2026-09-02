"""
Principal-component count study: pure-PCA two-point-statistics reconstruction
error versus the number of retained principal components.

For k = 1 .. 500, each held-out test sample is truncated to its first k PC
scores, inverse-transformed back to the two-point statistics field, and scored
with the per-sample normalized mean absolute error. The individual and
cumulative explained-variance ratios are recorded alongside. No PCA refit, no
model training. This reproduces Appendix C of the manuscript (Reviewer point
R2.1).

Run from the repository root:

    python scripts/06_pc_count_study.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import pickle

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from spinodal_cvae.metrics import mae

import config

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (4, 3),
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Palatino", "Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)

SPLIT = "test"  # held-out; the PCA basis is fit on the training split only
K_MAX = 500  # sweep upper bound
K_RETAINED = config.N_SCORES  # the choice this study characterizes (50)

OUTPUT_DIR = config.FIGURES_DIR / "pc_count_study"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------- load PCA ----
    # config.PATHS["pca"] is a plain sklearn PCA (see scripts/01_pca_on_stats.py),
    # holding all components. We never refit or mutate it: the k-sweep slices
    # components_ directly.
    with open(config.PATHS["pca"], "rb") as f:
        pca = pickle.load(f)

    evr_full = np.asarray(pca.explained_variance_ratio_)
    cum_evr_full = np.cumsum(evr_full)
    n_components_full = pca.components_.shape[0]
    k_max = min(K_MAX, n_components_full)
    print(f"PCA basis: {n_components_full} components (fit on the training split)")

    mean_ = pca.mean_.astype(np.float32)  # (n_features,)
    comps = pca.components_[:k_max].astype(np.float32)  # (k_max, n_features)

    # ------------------------------------------------- load 2PS test data ----
    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        stats = f["hetero_only"][SPLIT]["stats"][..., 0]  # (N, H, W); drop channel

    n_samples, h, w = stats.shape
    print(f"{SPLIT} split: {n_samples} samples, 2PS field {h}x{w}")
    assert h * w == comps.shape[1], "2PS field size does not match the PCA feature dim"

    X = stats.reshape(n_samples, -1).astype(np.float32)
    sample_std = np.std(stats, axis=(1, 2))  # Eq. (5) denominator (per sample)

    scores_full = (X - mean_) @ comps.T  # (N, k_max) — one projection

    # ------------------------------------------------------------ k-sweep ----
    k_grid = sorted(
        set(range(1, 65))
        | set(
            np.unique(
                np.round(np.logspace(np.log10(64), np.log10(k_max), 40)).astype(int)
            )
        )
        | {4, K_RETAINED, 100, 200, k_max}
    )
    k_grid = np.asarray([k for k in k_grid if 1 <= k <= k_max])

    nmae_mean = np.empty(len(k_grid))
    nmae_p10 = np.empty(len(k_grid))
    nmae_p90 = np.empty(len(k_grid))
    mae_mean = np.empty(len(k_grid))
    cum_ev = np.empty(len(k_grid))

    for j, k in enumerate(k_grid):
        recon = (scores_full[:, :k] @ comps[:k] + mean_).reshape(n_samples, h, w)
        mae_k = mae(stats, recon, axis=(1, 2))  # (N,)  Eq. (4)
        nmae_k = mae_k / sample_std  # (N,)  Eq. (5)
        mae_mean[j] = mae_k.mean()
        nmae_mean[j] = nmae_k.mean()
        nmae_p10[j] = np.percentile(nmae_k, 10)
        nmae_p90[j] = np.percentile(nmae_k, 90)
        cum_ev[j] = cum_evr_full[k - 1]

    evr_at_k = evr_full[k_grid - 1]

    # ------------------------------------------------------ summary table ----
    print(f"\nPC-count study — {SPLIT} split ({n_samples} samples), 2PS space\n")
    print(f"{'k':>5}  {'mean NMAE':>12}  {'mean MAE':>14}  {'cum. EV':>10}")
    for k in [4, 16, 32, K_RETAINED, 100, 200, k_max]:
        if k in k_grid:
            j = int(np.where(k_grid == k)[0][0])
            print(
                f"{k:>5}  {nmae_mean[j]:>12.4f}  {mae_mean[j]:>14.3e}  {cum_ev[j]:>10.5f}"
            )

    # ------------------------------------------------------- save arrays -----
    np.savez(
        OUTPUT_DIR / "pc_count_study.npz",
        split=SPLIT,
        n_samples=n_samples,
        k_grid=k_grid,
        nmae_mean=nmae_mean,
        nmae_p10=nmae_p10,
        nmae_p90=nmae_p90,
        mae_mean=mae_mean,
        cum_ev=cum_ev,
        evr_at_k=evr_at_k,
        evr_full=evr_full,
        cum_evr_full=cum_evr_full,
    )
    summary = {
        "split": SPLIT,
        "n_samples": int(n_samples),
        "k_retained": int(K_RETAINED),
        "k_max": int(k_max),
        "at_k": {
            str(k): {
                "nmae_mean": float(nmae_mean[int(np.where(k_grid == k)[0][0])]),
                "mae_mean": float(mae_mean[int(np.where(k_grid == k)[0][0])]),
                "cum_ev": float(cum_ev[int(np.where(k_grid == k)[0][0])]),
            }
            for k in [4, 16, 32, K_RETAINED, 100, 200, k_max]
            if k in k_grid
        },
    }
    with open(OUTPUT_DIR / "pc_count_study.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------------------------------------- figures ----
    def make_figure(xscale):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        (l_nmae,) = ax.plot(
            k_grid, nmae_mean, color="C0", marker="o", ms=3, label="2PS NMAE"
        )
        ax.fill_between(k_grid, nmae_p10, nmae_p90, color="C0", alpha=0.15, lw=0)
        ax.set_xscale(xscale)
        ax.set_xlabel("Number of retained principal components")
        ax.set_ylabel("2PS reconstruction NMAE", color="C0")
        ax.tick_params(axis="y", colors="C0")
        ax.set_xlim(1 if xscale == "log" else 0, k_max)

        ax2 = ax.twinx()
        (l_ev,) = ax2.plot(
            k_grid,
            evr_at_k,
            color="C3",
            ls="--",
            label="Individual explained-variance ratio",
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("Individual explained-variance ratio", color="C3")
        ax2.tick_params(axis="y", colors="C3")

        ax.legend(
            handles=[l_nmae, l_ev],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout()
        return fig

    for xscale, suffix in [("linear", "linear"), ("log", "log")]:
        fig = make_figure(xscale)
        out = OUTPUT_DIR / f"figure_pc_count_study_{suffix}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Figure saved to {out}")
