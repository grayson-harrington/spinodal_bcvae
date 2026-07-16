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
from sklearn.metrics import mean_absolute_error, pairwise_distances_argmin_min

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
    OUT_DIR = config.PATHS["figures_dir"] / "latent_manifold_2d_sweep"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_SCORES = config.N_SCORES

    # The two structurally informative latent dims, 0-indexed array positions (KL rank #1 and #2,
    # established by latent_traversal_2ps.py / latent_umap.py). Manuscript figures/text use 1-indexed
    # dimension numbers (e.g. main.tex: "dimensions two and twelve"; inference__kl_div_vs_latent_dim.png
    # x-axis runs 1..32) -- LATENT_DIM_X_LABEL/LATENT_DIM_Y_LABEL below carry that 1-indexed display
    # form so figure labels match the manuscript convention without renumbering the arrays.
    LATENT_DIM_X = 1
    LATENT_DIM_Y = 11
    LATENT_DIM_X_LABEL = LATENT_DIM_X + 1  # "2"
    LATENT_DIM_Y_LABEL = LATENT_DIM_Y + 1  # "12"

    GRID_SIZE = 6  # G x G thumbnail grid over (z_x, z_y)
    LATENT_RANGE = (-3, 3)

    SWEEP_PARAMS = ["X11", "mobility", "kappa"]  # the three parameters featured in Case Study II
    # (X11 = alloy composition x). Ordered composition/mobility/kappa to match Fig. 12 rows and
    # manuscript Fig. 15 panels (a)/(b)/(c).
    N_SWEEP_LEVELS = 3  # low / mid / high

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
        stats_test = f["hetero_only"]["test"]["stats"][..., 0]  # (N, H, W) ground-truth 2PS

    with h5py.File(config.PATHS["raw_data"], "r") as f:
        param_names = [s.decode("utf-8") for s in f["parameter_names"][...]]

    im_size = stats_test.shape[1]
    print(f"scores_test: {scores_test.shape}, params_test: {params_test.shape}")
    print(f"stats_test: {stats_test.shape}, im_size: {im_size}")

    # %% Encode test set to posterior-mean z (used throughout: KL ranking, ablation, scatter overlay)
    scores_scaled_test = scaler_scores.transform(scores_test)
    params_scaled_test = scaler_params.transform(params_test)
    z_mu_test, z_logvar_test = model.encode(X=scores_scaled_test, c=params_scaled_test)

    # %% Phase fraction / PC1-3 for axis-meaning-stability checks
    # NOTE: the existing latent-PC correlation heatmap (latent_pc_correlation.png) shows z1 and z11
    # correlate strongly with PC2 and PC3 (r~0.7-0.8), NOT PC1 (phase fraction) -- confirmed before
    # building the axis-meaning check below, rather than assumed.
    phase_fraction = stats_test.reshape(stats_test.shape[0], -1).mean(axis=1)
    pc1 = scores_test[:, 0]
    pc2 = scores_test[:, 1]
    pc3 = scores_test[:, 2]

    # ============================================================================
    # Part A -- Effective dimensionality quantification
    # ============================================================================

    # %% Per-dimension KL divergence and cumulative excess-KL scree plot
    #
    # Every dim in a beta-VAE carries a nonzero KL floor even when uninformative (a diffuse-encoding
    # regime distinct from posterior collapse), so raw KL/participation-ratio conflates "shared floor"
    # with "active dimension" -- e.g. raw PR here is a misleading ~29/32 despite two dims visibly
    # dominating in the manuscript's own KL-vs-dim plot. Only variation *above* the shared floor is
    # informative, so we subtract the floor (median KL of the non-spiking dims) and report the
    # cumulative excess-KL mass directly, with plain top-k readouts -- no single summary statistic
    # (e.g. participation ratio) to defend, just the curve and the numbers it implies.
    kl = -0.5 * (1 + z_logvar_test - z_mu_test**2 - np.exp(z_logvar_test)).mean(axis=0)
    top_idx = np.argsort(kl)[::-1]

    baseline_kl = np.median(kl[top_idx[2:]])  # shared floor, excluding the two candidate spikes
    excess_kl = np.clip(kl - baseline_kl, 0, None)

    print(f"Per-dim KL (descending): {kl[top_idx].round(4)}")
    print(f"Baseline KL floor (median, excl. top 2): {baseline_kl:.4f}")

    cum_excess_frac = np.cumsum(excess_kl[top_idx]) / excess_kl.sum()
    for k in (1, 2, 5, 10):
        print(f"Top {k:2d} dims: {cum_excess_frac[k - 1] * 100:5.1f}% of excess KL mass")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(
        np.arange(1, latent_dim + 1), cum_excess_frac, marker="s", markersize=4,
        color="firebrick",
    )
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.axvline(2, color="gray", linewidth=0.8, linestyle=":", label="2 dims")
    ax.set_xlabel("Number of latent dims (sorted by excess KL, descending)", fontsize=13)
    ax.set_ylabel("Cumulative fraction of excess KL\n(above shared floor)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cumulative_kl_scree.png", bbox_inches="tight")
    plt.show()

    # %% 2-D-only ablation: decode using only dims {LATENT_DIM_X, LATENT_DIM_Y}, rest fixed at prior
    # mean (0), and compare reconstruction MAE against the full 32-D decode and against ground truth.

    def decode_2ps(z, c):
        decoded_scaled = model.decode(z=z, c=c)
        decoded_scores = scaler_scores.inverse_transform(decoded_scaled)
        stats = pca.inverse_transform(decoded_scores)
        return stats.reshape(-1, im_size, im_size)

    z_full = z_mu_test.astype(np.float32)
    z_ablated = np.zeros_like(z_full)
    z_ablated[:, LATENT_DIM_X] = z_full[:, LATENT_DIM_X]
    z_ablated[:, LATENT_DIM_Y] = z_full[:, LATENT_DIM_Y]

    stats_full_decoded = decode_2ps(z_full, params_scaled_test.astype(np.float32))
    stats_ablated_decoded = decode_2ps(z_ablated, params_scaled_test.astype(np.float32))

    mae_full_vs_true = np.array(
        [
            mean_absolute_error(stats_test[i].ravel(), stats_full_decoded[i].ravel())
            for i in range(len(stats_test))
        ]
    )
    mae_2d_vs_true = np.array(
        [
            mean_absolute_error(stats_test[i].ravel(), stats_ablated_decoded[i].ravel())
            for i in range(len(stats_test))
        ]
    )
    mae_2d_vs_full = np.array(
        [
            mean_absolute_error(stats_full_decoded[i].ravel(), stats_ablated_decoded[i].ravel())
            for i in range(len(stats_test))
        ]
    )

    print("\n=== Reconstruction MAE summary (2PS field, full test set) ===")
    print(f"{'':30s} {'mean':>10s} {'median':>10s}")
    print(f"{'full 32-D z vs. ground truth':30s} {mae_full_vs_true.mean():10.4f} {np.median(mae_full_vs_true):10.4f}")
    print(f"{'2-D-only z vs. ground truth':30s} {mae_2d_vs_true.mean():10.4f} {np.median(mae_2d_vs_true):10.4f}")
    print(f"{'2-D-only z vs. full 32-D decode':30s} {mae_2d_vs_full.mean():10.4f} {np.median(mae_2d_vs_full):10.4f}")

    ablation_summary_path = OUT_DIR / "ablation_mae_summary.txt"
    with open(ablation_summary_path, "w") as f:
        f.write(f"baseline_kl_floor\t{baseline_kl:.4f}\n")
        for k in (1, 2, 5, 10):
            f.write(f"excess_kl_frac_top{k}\t{cum_excess_frac[k - 1]:.4f}\n")
        f.write(f"full32D_vs_truth_mean\t{mae_full_vs_true.mean():.4f}\n")
        f.write(f"full32D_vs_truth_median\t{np.median(mae_full_vs_true):.4f}\n")
        f.write(f"2Donly_vs_truth_mean\t{mae_2d_vs_true.mean():.4f}\n")
        f.write(f"2Donly_vs_truth_median\t{np.median(mae_2d_vs_true):.4f}\n")
        f.write(f"2Donly_vs_full32D_mean\t{mae_2d_vs_full.mean():.4f}\n")
        f.write(f"2Donly_vs_full32D_median\t{np.median(mae_2d_vs_full):.4f}\n")
    print(f"Saved -> {ablation_summary_path}")

    # ============================================================================
    # Part B -- Direct (z_x, z_y) decoded grid, repeated across one-parameter sweeps
    # ============================================================================

    # %% Base representative condition (KMeans + closest-to-centroid, consistent with
    # latent_traversal_2ps.py / Case Study II selection convention), shared across all sweeps below.
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(params_test)
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, params_test)
    base_test_idx = closest_indices[0]
    base_param = params_test[base_test_idx].copy()

    grid_1d = np.linspace(*LATENT_RANGE, GRID_SIZE, dtype=np.float32)
    ZX, ZY = np.meshgrid(grid_1d, grid_1d)  # (G, G) each

    def decode_sweep_grid(sweep_param: str):
        """Decode the (z_x, z_y) grid at low/mid/high `sweep_param`. Returns (sweep_values,
        decoded_grids) where decoded_grids[k] has shape (GRID_SIZE, GRID_SIZE, im_size, im_size)."""
        sweep_idx = param_names.index(sweep_param)
        sweep_values = np.quantile(params_test[:, sweep_idx], np.linspace(0.1, 0.9, N_SWEEP_LEVELS))
        print(f"\n{sweep_param} sweep values (low/mid/high): {sweep_values}")

        decoded_grids = []
        for sweep_val in sweep_values:
            cond = base_param.copy()
            cond[sweep_idx] = sweep_val
            cond_scaled = scaler_params.transform(cond[None, :])[0].astype(np.float32)

            Z_grid = np.zeros((GRID_SIZE * GRID_SIZE, latent_dim), dtype=np.float32)
            Z_grid[:, LATENT_DIM_X] = ZX.ravel()
            Z_grid[:, LATENT_DIM_Y] = ZY.ravel()
            C_grid = np.tile(cond_scaled, (GRID_SIZE * GRID_SIZE, 1))

            stats_grid = decode_2ps(Z_grid, C_grid).reshape(GRID_SIZE, GRID_SIZE, im_size, im_size)
            decoded_grids.append(stats_grid)

        return sweep_values, decoded_grids

    def stitch_mosaic(stats_grid, sep_px: int = 2, sep_value: float = np.nan):
        """Stitch a (GRID_SIZE, GRID_SIZE, im_size, im_size) array of thumbnails into one
        single 2D mosaic image, with a `sep_px`-wide separator of `sep_value` between every
        thumbnail in both directions (equal spacing, guaranteed pixel-exact -- avoids subplot-
        grid rendering artifacts from stitching many separate imshow() axes)."""
        G = stats_grid.shape[0]
        tile = im_size
        mosaic_size = G * tile + (G - 1) * sep_px
        mosaic = np.full((mosaic_size, mosaic_size), sep_value, dtype=np.float32)
        for i in range(G):  # row index, increasing z_y bottom-to-top
            row = G - 1 - i
            y0 = row * (tile + sep_px)
            for j in range(G):  # column index, increasing z_x left-to-right
                x0 = j * (tile + sep_px)
                mosaic[y0 : y0 + tile, x0 : x0 + tile] = stats_grid[i, j]
        return mosaic

    def render_sweep_grid(sweep_param, sweep_values, decoded_grids, vmin, vmax):
        """Render all sweep levels side by side as single stitched-mosaic images, sharing
        global x/y axes labeled in actual (1-indexed) latent position, and one shared colorbar
        on the right using the same (vmin, vmax) passed in across every call -- so brightness is
        directly comparable across sweep params, not just within one."""
        panel_size = 4.2  # inches; each level's panel is exactly square (equal width and height)
        fig, axs = plt.subplots(
            1, N_SWEEP_LEVELS, figsize=(panel_size * N_SWEEP_LEVELS, panel_size), squeeze=False,
        )
        axs = axs[0]
        tick_pos = np.linspace(0, GRID_SIZE - 1, 3) / (GRID_SIZE - 1)
        tick_lab = [f"{v:.0f}" for v in np.linspace(*LATENT_RANGE, 3)]
        cmap = mpl.colormaps["inferno"].copy()
        cmap.set_bad(color="white")  # separator pixels (NaN) render as white, same in both directions

        # Compact qualitative level tags (Low/Mid/High). No numeric embedded title -- the actual
        # sweep values and the parameter name are reported in the manuscript caption instead.
        level_tags = ["Low", "Mid", "High"]

        im = None
        for level_i, (sweep_val, stats_grid) in enumerate(zip(sweep_values, decoded_grids)):
            mosaic = stitch_mosaic(stats_grid)
            ax = axs[level_i]
            im = ax.imshow(mosaic, cmap=cmap, vmin=vmin, vmax=vmax, extent=(0, 1, 0, 1))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lab)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lab)
            ax.set_xlabel(f"$z_{{{LATENT_DIM_X_LABEL}}}$")
            ax.set_ylabel(f"$z_{{{LATENT_DIM_Y_LABEL}}}$")
            ax.text(
                0.03, 0.97, level_tags[level_i], transform=ax.transAxes,
                ha="left", va="top", fontweight="bold", color="white", fontsize=10,
            )
            ax.tick_params(length=3)

        fig.colorbar(im, ax=axs, label="2PS intensity", fraction=0.025, pad=0.02)
        fig.savefig(OUT_DIR / f"grid_{sweep_param}_all_levels.png")
        plt.show()

    def axis_meaning_check(sweep_param: str, sweep_values):
        """Axis-meaning stability: does z_x/z_y keep the same physical interpretation across
        conditions, or does the mapping drift? For real test points near each sweep condition,
        correlate z_x/z_y with PC2 and PC3 (the PCs they actually correlate with, per the existing
        latent-PC correlation heatmap -- not PC1/phase fraction, which they do not track), and
        compare across levels. A stable correlation across conditions supports "z_x/z_y are a
        consistent physical coordinate system"; a level-dependent correlation would mean the axes'
        meaning is condition-dependent -- an honest limitation worth reporting either way."""
        sweep_idx = param_names.index(sweep_param)
        tol = (params_test[:, sweep_idx].max() - params_test[:, sweep_idx].min()) * 0.1

        axis_meaning_rows = [
            ["level", sweep_param, "n", "corr(zx,PC2)", "corr(zx,PC3)", "corr(zy,PC2)", "corr(zy,PC3)"]
        ]
        for level_i, sweep_val in enumerate(sweep_values):
            near_mask = np.abs(params_test[:, sweep_idx] - sweep_val) < tol
            z_near = z_mu_test[near_mask]

            r_zx_pc2 = np.corrcoef(z_near[:, LATENT_DIM_X], pc2[near_mask])[0, 1]
            r_zx_pc3 = np.corrcoef(z_near[:, LATENT_DIM_X], pc3[near_mask])[0, 1]
            r_zy_pc2 = np.corrcoef(z_near[:, LATENT_DIM_Y], pc2[near_mask])[0, 1]
            r_zy_pc3 = np.corrcoef(z_near[:, LATENT_DIM_Y], pc3[near_mask])[0, 1]

            axis_meaning_rows.append(
                [
                    str(level_i), f"{sweep_val:.3e}", str(near_mask.sum()),
                    f"{r_zx_pc2:.3f}", f"{r_zx_pc3:.3f}", f"{r_zy_pc2:.3f}", f"{r_zy_pc3:.3f}",
                ]
            )

        # whole-test-set reference row (n=171), to judge whether subgroup correlations (n~30 each,
        # noisier) are consistent with the full-sample relationship or genuinely drifting
        r_zx_pc2_all = np.corrcoef(z_mu_test[:, LATENT_DIM_X], pc2)[0, 1]
        r_zx_pc3_all = np.corrcoef(z_mu_test[:, LATENT_DIM_X], pc3)[0, 1]
        r_zy_pc2_all = np.corrcoef(z_mu_test[:, LATENT_DIM_Y], pc2)[0, 1]
        r_zy_pc3_all = np.corrcoef(z_mu_test[:, LATENT_DIM_Y], pc3)[0, 1]
        axis_meaning_rows.append(
            [
                "all", "-", str(len(z_mu_test)),
                f"{r_zx_pc2_all:.3f}", f"{r_zx_pc3_all:.3f}", f"{r_zy_pc2_all:.3f}", f"{r_zy_pc3_all:.3f}",
            ]
        )

        col_widths = [max(len(row[c]) for row in axis_meaning_rows) for c in range(len(axis_meaning_rows[0]))]
        axis_meaning_str = "\n".join(
            "  ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) for row in axis_meaning_rows
        )
        print(f"\n=== Axis-meaning stability across {sweep_param} levels ===")
        print(axis_meaning_str)

        axis_meaning_path = OUT_DIR / f"axis_meaning_stability_{sweep_param}.txt"
        with open(axis_meaning_path, "w") as f:
            f.write(axis_meaning_str + "\n")
        print(f"Saved -> {axis_meaning_path}")

    # %% Decode every sweep first. Each sweep param gets its own color scale (shared only
    # across its own low/mid/high levels) so weaker-contrast sweeps (e.g. composition) aren't
    # washed out by a scale set by a higher-contrast sweep.
    sweep_results = {}
    for sweep_param in SWEEP_PARAMS:
        sweep_values, decoded_grids = decode_sweep_grid(sweep_param)
        sweep_results[sweep_param] = (sweep_values, decoded_grids)

    # %% Render each sweep's grid figure (own color scale per sweep param) and its axis-meaning table
    for sweep_param, (sweep_values, decoded_grids) in sweep_results.items():
        vmin = min(stats_grid.min() for stats_grid in decoded_grids)
        vmax = max(stats_grid.max() for stats_grid in decoded_grids)
        render_sweep_grid(sweep_param, sweep_values, decoded_grids, vmin, vmax)
        axis_meaning_check(sweep_param, sweep_values)

    # %%
    print("\nDone. Inspect figures/tables under:", OUT_DIR)
