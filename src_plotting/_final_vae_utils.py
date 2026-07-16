from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np

from spinodal_cvae.metrics import mae

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config

config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

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

latex_labels = {
    "X11": r"$x$",
    "V_m": r"$V_m^{Mg_2Sn}$",
    "V_p": r"$V_m^{Mg_2Si}$",
    "eps_T": r"$\varepsilon^T$",
    "C11_m": r"$C_{11}^{Mg_2Sn}$",
    "C12_m": r"$C_{12}^{Mg_2Sn}$",
    "C44_m": r"$C_{44}^{Mg_2Sn}$",
    "C11_p": r"$C_{11}^{Mg_2Si}$",
    "C12_p": r"$C_{12}^{Mg_2Si}$",
    "C44_p": r"$C_{44}^{Mg_2Si}$",
    "mobility": r"$M$",
    "kappa": r"$\kappa$",
    "L0_Si_Sn": r"$^0a^{ss}$",
    "L1_Si_Sn": r"$^0b^{ss}$",
    "L2_Si_Sn": r"$^1a^{ss}$",
    "L0_Si_Sn_liq": r"$^0a^{liq}$",
    "L1_Si_Sn_liq": r"$^0b^{liq}$",
    "L2_Si_Sn_liq": r"$^1a^{liq}$",
}


def best_grid(n_pcs):
    """
    Find subplot grid (nx, ny) such that nx * ny == n_pcs,
    with preference for horizontal width (nx >= ny).
    If n_pcs is prime, we allow empty slots by using (ceil, ceil).
    """
    best = None
    for ny in range(1, int(math.sqrt(n_pcs)) + 1):
        if n_pcs % ny == 0:
            nx = n_pcs // ny
            if best is None or nx >= ny and nx - ny < best[0] - best[1]:
                best = (nx, ny)

    if best is None:
        # fallback for prime numbers: rectangle with empty slots
        nx = math.ceil(math.sqrt(n_pcs))
        ny = math.ceil(n_pcs / nx)
        best = (nx, ny)

    return best


def plot_losses(losses, show_plot=True):
    plt.figure(figsize=(7, 3))

    plt.subplot(121)
    plt.plot(losses["train_recon"], label="Train")
    plt.plot(losses["val_recon"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend()

    plt.subplot(122)
    plt.plot(losses["train_kl"])
    plt.plot(losses["val_kl"])
    plt.xlabel("Epoch")
    plt.ylabel(r"$\beta$-scaled KL Divergence")

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig(config.PATHS["figures_dir"] / "losses.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_parity_plots(
    scores_train,
    scores_train_recon,
    scores_validate,
    scores_validate_recon,
    n_pcs=16,
    show_plot=True,
):
    # pick subplot grid ~square
    nx, ny = best_grid(n_pcs)

    plt.figure(figsize=(2.5 * nx, 2.5 * ny))  # dynamic figure size

    for i in range(n_pcs):
        if i >= scores_train.shape[1]:
            break  # Don't try to plot more PCs than you have
        plt.subplot(ny, nx, i + 1)
        plt.scatter(
            scores_train[:, i], scores_train_recon[:, i], alpha=0.5, label="Train"
        )
        plt.scatter(
            scores_validate[:, i],
            scores_validate_recon[:, i],
            alpha=0.5,
            label="Test",
        )
        # Parity line
        min_val = min(scores_train[:, i].min(), scores_validate[:, i].min())
        max_val = max(scores_train[:, i].max(), scores_validate[:, i].max())
        diff = max_val - min_val
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            lw=2,
        )
        plt.xlim(min_val - diff / 10, max_val + diff / 10)
        plt.ylim(min_val - diff / 10, max_val + diff / 10)
        plt.xlabel(f"PC{i + 1} True")
        plt.ylabel(f"PC{i + 1} Predicted")
        if i == 0:
            plt.legend()

    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.3, hspace=0.4
    )

    fig = plt.gcf()
    fig.savefig(config.PATHS["figures_dir"] / f"parity_plots_{n_pcs}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_relative_MAE_per_score(
    scores_train: np.ndarray,
    scores_train_recon: np.ndarray,
    scores_test: np.ndarray,
    scores_test_recon: np.ndarray,
):
    # Compute range (max - min) per PC from training set
    pc_min = np.min(scores_train, axis=0)
    pc_max = np.max(scores_train, axis=0)
    pc_range = pc_max - pc_min

    # Calculate Relative MAE per PC (as a fraction of range)
    rel_mae_train = []
    rel_mae_test = []

    n_pcs = scores_train.shape[-1]

    for i in range(n_pcs):
        mae_train_i = mae(scores_train[:, i], scores_train_recon[:, i], axis=0)
        mae_val_i = mae(scores_test[:, i], scores_test_recon[:, i], axis=0)

        rel_mae_train.append(
            mae_train_i / (pc_range[i] + 1e-8)
        )  # avoid division by zero
        rel_mae_test.append(mae_val_i / (pc_range[i] + 1e-8))

    rel_mae_train = np.array(rel_mae_train)
    rel_mae_test = np.array(rel_mae_test)

    return rel_mae_train, rel_mae_test


def plot_relative_MAE_vs_score(
    rel_mae_train: np.ndarray,
    rel_mae_test: np.ndarray,
    show_plot=True,
):
    rel_mae_train_pct = rel_mae_train
    rel_mae_test_pct = rel_mae_test

    # Moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    window = 3
    x = np.arange(1, len(rel_mae_train) + 1)
    ma_x = np.arange(window // 2 + 1, len(rel_mae_train) - window // 2 + 1)

    # Plot
    plt.figure(figsize=(5, 3.5))
    plt.scatter(
        x,
        rel_mae_train_pct,
        label="Train RMAE",
        color="tab:blue",
        alpha=0.6,
    )
    plt.scatter(
        x,
        rel_mae_test_pct,
        label="Test RMAE",
        color="tab:orange",
        alpha=0.6,
    )

    plt.plot(
        ma_x,
        moving_average(rel_mae_train_pct, window),
        color="tab:blue",
        linewidth=2,
        label="Train Moving Avg",
    )
    plt.plot(
        ma_x,
        moving_average(rel_mae_test_pct, window),
        color="tab:orange",
        linewidth=2,
        label="Test Moving Avg",
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Reconstruction Error (RMAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig(config.PATHS["figures_dir"] / "relative_mae_vs_pc_scores.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return rel_mae_train, rel_mae_test


def get_KL_vs_latent_dim(z_mu_train, z_logvar_train, z_mu_test, z_logvar_test):
    """
    Compute the true analytical KL divergence per latent dimension for both
    training and test sets.

    Returns
    -------
    kl_train : np.ndarray
        Average KL divergence per latent dim for training set.
    kl_test : np.ndarray
        Average KL divergence per latent dim for test set.
    """

    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu**2 - np.exp(logvar))  # shape: [N, D]

    kl_train = kl_div(z_mu_train, z_logvar_train).mean(axis=0)
    kl_test = kl_div(z_mu_test, z_logvar_test).mean(axis=0)

    return kl_train, kl_test


def plot_KL_vs_latent_dim(kl_train, kl_test, show_plot=True):
    x = np.arange(1, len(kl_train) + 1)

    plt.figure(figsize=(6, 4))

    # plot averages first
    avg_kl_train = kl_train.mean()
    avg_kl_test = kl_test.mean()
    plt.axhline(
        avg_kl_train,
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label=f"Train Avg:   {avg_kl_train:.3f}",
    )
    plt.axhline(
        avg_kl_test,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=f"Test Avg:     {avg_kl_test:.3f}",
    )

    # plot raw points and connections second
    plt.scatter(x, kl_train, color="tab:blue", label="Train KL Divergence")
    plt.scatter(x, kl_test, color="tab:orange", label="Test KL Divergence")
    plt.plot(x, kl_train, color="tab:blue", alpha=0.5)
    plt.plot(x, kl_test, color="tab:orange", alpha=0.5)

    plt.xlabel("Latent Dimension")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()
    fig.savefig(config.PATHS["figures_dir"] / "kl_div_vs_latent_dim.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_distribution_fidelity_pc_histograms(
    reference_pc: np.ndarray,
    reconstructed_pc: np.ndarray,
    generated_pc: np.ndarray,
    show_plot: bool = True,
    label_reference: str = "Train",
    label_reconstructed: str = "Reconstruction (Posterior)",
    label_generated: str = "Generation (Prior)",
    n_pcs: int = 8,
    bins: int = 30,
):
    """
    Overlaid histograms comparing marginal PC distributions between:
      1) empirical reference data,
      2) posterior predictive reconstructions,
      3) prior-sampled generative outputs.

    Parameters
    ----------
    reference_pc : (N_ref, n_scores)
        Empirical PC scores (typically training data).
    reconstructed_pc : (N_rec, n_scores)
        Posterior predictive samples (encode -> sample -> decode).
    generated_pc : (N_gen, n_scores)
        Generative samples (prior -> decode).
    """
    nx, ny = best_grid(n_pcs)

    fig, axs = plt.subplots(ny, nx, figsize=(2.5 * nx, 2.5 * ny))
    axs = axs.flatten()

    for i in range(n_pcs):
        # Reference (empirical)
        axs[i].hist(
            reference_pc[:, i],
            bins=bins,
            # histtype="step",
            # linestyle="-.",
            # linewidth=1.0,
            alpha=0.65,
            color="tab:blue",
            density=True,
            label=label_reference,
            # zorder=-10,
        )

        # Posterior predictive (reconstruction)
        axs[i].hist(
            reconstructed_pc[:, i],
            bins=bins,
            alpha=0.65,
            color="tab:orange",
            density=True,
            label=label_reconstructed,
        )

        # Generative (prior-sampled)
        axs[i].hist(
            generated_pc[:, i],
            bins=bins,
            histtype="step",
            linestyle="--",
            linewidth=1.0,
            color="black",
            density=True,
            label=label_generated,
        )

        axs[i].set_xlabel(f"PC{i + 1}")
        axs[i].set_ylabel("Probability Density")
        axs[i].tick_params(axis="x", labelrotation=45)

    # Single shared legend
    axs[3].legend()
    plt.tight_layout(h_pad=2.0)

    fig.savefig(config.PATHS["figures_dir"] / "distribution_fidelity_pc_histograms.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def pick_example_indices(errors: np.ndarray, n_mean: int = 3):
    idx_min = int(np.argmin(errors))
    idx_max = int(np.argmax(errors))

    mean_val = float(np.mean(errors))
    # sort indices by distance to mean, exclude min and max
    candidate_indices = np.argsort(np.abs(errors - mean_val))
    idx_means = [int(i) for i in candidate_indices if i not in [idx_min, idx_max]][
        :n_mean
    ]

    return [idx_min] + idx_means + [idx_max]


def plot_selected_recon_examples(
    stats: np.ndarray,
    stats_trunc: np.ndarray,
    stats_sampl: np.ndarray,
    indices: list,
    show_plot: bool = True,
):
    """
    Columns:
        Original | Truncated | Sampled | Stats CBar | (spacer) | Error | Error CBar
    """

    from matplotlib.gridspec import GridSpec

    col_titles = ["Original", "Truncated", "Sampled", "|Original - Sampled|"]

    row_labels = [
        "Min Error",
        "Near Mean Error",
        "Near Mean Error",
        "Near Mean Error",
        "Max Error",
    ]

    # --- Global error scale across all selected examples ---
    all_errs = [np.abs(stats[i] - stats_sampl[i]) for i in indices]
    global_err_vmin = 0.0
    global_err_vmax = float(max(err.max() for err in all_errs))

    for plot_idx, i in enumerate(indices):
        og = stats[i]
        tr = stats_trunc[i]
        re = stats_sampl[i]
        err = np.abs(og - re)

        # Row-local stats scale
        stat_vmin = float(min(og.min(), tr.min(), re.min()))
        stat_vmax = float(max(og.max(), tr.max(), re.max()))

        fig = plt.figure(figsize=(9, 2.25))
        gs = GridSpec(
            1,
            7,
            figure=fig,
            width_ratios=[1, 1, 1, 0.10, 0.4, 1, 0.10],
            wspace=0.1,
        )

        # Image axes
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_trunc = fig.add_subplot(gs[0, 1])
        ax_samp = fig.add_subplot(gs[0, 2])
        ax_err = fig.add_subplot(gs[0, 5])

        # --- Original ---
        im_stat = ax_orig.imshow(og, cmap="viridis", vmin=stat_vmin, vmax=stat_vmax)
        ax_orig.set_ylabel(row_labels[plot_idx], fontsize=11)
        ax_orig.set_title(col_titles[0], fontsize=12)

        # --- Truncated ---
        ax_trunc.imshow(tr, cmap="viridis", vmin=stat_vmin, vmax=stat_vmax)
        ax_trunc.set_title(col_titles[1], fontsize=12)

        # --- Sampled ---
        ax_samp.imshow(re, cmap="viridis", vmin=stat_vmin, vmax=stat_vmax)
        ax_samp.set_title(col_titles[2], fontsize=12)

        # --- Error ---
        im_err = ax_err.imshow(
            err, cmap="inferno", vmin=global_err_vmin, vmax=global_err_vmax
        )
        ax_err.set_title(col_titles[3], fontsize=12)

        for ax in (ax_orig, ax_trunc, ax_samp, ax_err):
            ax.set_xticks([])
            ax.set_yticks([])

        # --- Colorbars ---
        cax_stat = fig.add_subplot(gs[0, 3])
        cax_err = fig.add_subplot(gs[0, 6])

        cb_stat = plt.colorbar(im_stat, cax=cax_stat)
        cb_stat.ax.tick_params(labelsize=8)

        cb_err = plt.colorbar(im_err, cax=cax_err)
        cb_err.ax.tick_params(labelsize=8)

        fig.tight_layout()

        fig.savefig(
            config.PATHS["figures_dir"] / f"reconstruction_example_{plot_idx + 1}.png"
        )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_vae_sample_cross_sections(stats_test_img, stats_vae_img, show_plot=True):
    sample_labels = [
        "Min Error",
        "Near Mean Error",
        "Near Mean Error",
        "Near Mean Error",
        "Max Error",
    ]

    n_test, n_vae, im_size, _ = stats_vae_img.shape
    center_row = im_size // 2

    for i in range(n_test):
        fig, ax = plt.subplots(figsize=(6, 2.25))

        vae_cross_sections = stats_vae_img[i, :, center_row, :]

        # Plot all VAE sample cross-sections (gray)
        for j in range(n_vae):
            ax.plot(
                vae_cross_sections[j],
                color="gray",
                alpha=0.1,
                linewidth=0.5,
            )

        # Mean VAE cross-section (closest sample)
        mean_cross_section = vae_cross_sections.mean(axis=0)
        ind_mean = np.argmin(
            np.abs(vae_cross_sections - mean_cross_section).sum(axis=1)
        )
        ax.plot(
            vae_cross_sections[ind_mean],
            label="Mean Sample",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Closest to original
        ind_closest = np.argmin(
            np.abs(vae_cross_sections - stats_test_img[i, center_row, :]).sum(axis=1)
        )
        ax.plot(
            vae_cross_sections[ind_closest],
            label="Closest Sample",
            color="blue",
            linestyle="--",
            linewidth=2,
        )

        # Original cross-section
        ax.plot(
            stats_test_img[i, center_row, :],
            label="Original",
            linewidth=2,
            color="black",
        )

        # Y-axis limits with padding
        data_min = np.min(vae_cross_sections)
        data_max = np.max(vae_cross_sections)
        pad = 0.05 * (data_max - data_min)
        ax.set_ylim(data_min - pad, data_max + pad)

        ax.set_title(sample_labels[i])
        ax.set_xlabel("Pixel Index in Middle Cross-Section")
        ax.set_ylabel("Correlation")
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 1]
        ax.legend(
            [handles[k] for k in order],
            [labels[k] for k in order],
        )

        fig.tight_layout()

        fig.savefig(config.PATHS["figures_dir"] / f"cross_section_{i + 1}.png")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_recon_pc_histograms(
    scores_train: np.ndarray,
    scores_selected: np.ndarray,
    vae_samples_unscaled: np.ndarray,
    show_plot: bool = True,
    n_pcs: int = 5,
    bins: int = 40,
):
    """
    Plot per-PC histograms instead of rug plots.

    - scores_train: shape (n_train, n_pcs)
    - scores_selected: shape (n_samples, n_pcs)
    - vae_samples_unscaled: shape (n_samples, n_vae_samples, n_pcs)
    """
    n_samples = vae_samples_unscaled.shape[0]

    sample_labels = [
        "Min Error",
        "Near Mean Error",
        "Near Mean Error",
        "Near Mean Error",
        "Max Error",
    ]

    nx, ny = best_grid(n_pcs)

    for i in range(n_samples):
        fig, axs = plt.subplots(ny, nx, figsize=(3 * nx, 3 * ny), sharey=False)
        axs = axs.flatten()
        fig.suptitle(sample_labels[i], fontsize=16)

        for j in range(n_pcs):
            ax = axs[j]

            # Histogram: training PC distribution
            ax.hist(
                scores_train[:, j],
                bins=bins,
                density=True,
                alpha=0.5,
                color="tab:blue",
                label="PC Distribution" if j == 0 else None,
            )

            # Histogram: VAE samples
            ax.hist(
                vae_samples_unscaled[i, :, j],
                bins=bins,
                density=True,
                alpha=0.5,
                color="tab:orange",
                label="VAE Sample Distribution" if j == 0 else None,
            )

            # Vertical line: original/test sample
            ax.axvline(
                scores_selected[i, j],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Test Sample Value" if j == 0 else None,
            )

            ax.set_xlabel(f"PC{j + 1}")
            ax.tick_params(axis="x", labelrotation=45)

        # Opaque legend handles
        legend_handles = [
            Line2D(
                [0], [0], color="red", lw=4, linestyle="--", label="Test Sample Value"
            ),
            Line2D([0], [0], color="tab:blue", lw=4, label="PC Distribution"),
            Line2D([0], [0], color="tab:orange", lw=4, label="VAE Samples"),
        ]
        axs[0].legend(handles=legend_handles, loc="upper left")

        plt.tight_layout()

        fig.savefig(config.PATHS["figures_dir"] / f"recon_pc_histograms_{i + 1}.png")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def sample_cvae_in_batches(model, total_samples: int, c: np.ndarray, batch_size: int):
    """
    Calls model.sample in mini-batches; c must be (total_samples, cond_dim)
    Returns (total_samples, n_scores) in model's scaled space.
    """
    outs = []
    for s in range(0, total_samples, batch_size):
        e = min(s + batch_size, total_samples)
        outs.append(model.sample(num_samples=e - s, c=c[s:e]))
    return np.vstack(outs)


def sweep_one_condition(
    cond_idx: int,
    base_param: np.ndarray,
    params_train: np.ndarray,
    params_scaler,
    score_scaler,
    model,
    n_sweep: int = 100,
    n_per_cond: int = 5000,
    batch_size: int = 20000,
):
    """
    Sweep a single condition dimension cond_idx while holding others at base_param.
    Returns:
      sweep_values: (n_sweep,)
      mean_pc: (n_sweep, n_scores)
      std_pc:  (n_sweep, n_scores)
    """
    # Range for this condition from training data (unscaled space)
    vmin = params_train[:, cond_idx].min()
    vmax = params_train[:, cond_idx].max()
    sweep_values = np.linspace(vmin, vmax, n_sweep)

    # Build unscaled condition matrix to sweep only cond_idx
    varied_params = np.tile(base_param, (n_sweep, 1))
    varied_params[:, cond_idx] = sweep_values

    # Repeat each condition n_per_cond times -> (n_sweep * n_per_cond, cond_dim)
    varied_params_expanded = np.repeat(varied_params, n_per_cond, axis=0)

    # Scale conditions once
    varied_params_scaled = params_scaler.transform(varied_params_expanded)

    # Sample from CVAE in batches
    total = n_sweep * n_per_cond
    sampled_scores_scaled = sample_cvae_in_batches(
        model=model,
        total_samples=total,
        c=varied_params_scaled,
        batch_size=batch_size,
    )

    # Inverse-transform scores and reshape to (n_sweep, n_per_cond, n_scores)
    sampled_scores = score_scaler.inverse_transform(sampled_scores_scaled)
    sampled_scores = sampled_scores.reshape(n_sweep, n_per_cond, -1)

    # Aggregate
    mean_pc = sampled_scores.mean(axis=1)  # (n_sweep, n_scores)
    std_pc = sampled_scores.std(axis=1)  # (n_sweep, n_scores)

    return sweep_values, mean_pc, std_pc


def plot_pc_trend_with_variance_multi(
    x_values: np.ndarray,
    mean_pcs: np.ndarray,
    std_pcs: np.ndarray,
    title: str,
    x_label: str = "Condition value",
    show_plot: bool = True,
    n_pcs: int = 8,
):
    nx, ny = best_grid(n_pcs)

    fig, axs = plt.subplots(ny, nx, figsize=(3 * nx, 3 * ny), sharey=False)
    axs = axs.flatten()

    n_samples = mean_pcs.shape[0]

    for pc in range(n_pcs):
        ax = axs[pc]
        for s in range(n_samples):
            m = mean_pcs[s, :, pc]
            std = std_pcs[s, :, pc]

            ax.plot(
                x_values,
                m,
                # alpha=0.9,
                label=f"Sample {chr(ord('A') + s)}" if pc == 0 else None,
            )
            ax.fill_between(
                x_values,
                m - std,
                m + std,
                alpha=0.15,
            )

        ax.set_ylabel(f"PC{pc + 1}")
        ax.set_xlabel(latex_labels[x_label])

    axs[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()

    fig.savefig(config.PATHS["figures_dir"] / f"{title.lower().replace(' ', '_')}.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
