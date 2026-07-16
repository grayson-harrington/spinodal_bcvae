import h5py
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config


def load_hetero_only_data(path_data, n_scores):
    with h5py.File(path_data, "r") as f:
        scores_train = f["hetero_only"]["train"]["scores"][:, :n_scores]
        params_train = f["hetero_only"]["train"]["params"][...]

        scores_validate = f["hetero_only"]["validate"]["scores"][:, :n_scores]
        params_validate = f["hetero_only"]["validate"]["params"][...]

        print("Train")
        print(scores_train.shape, params_train.shape)

        print("Validate")
        print(scores_validate.shape, params_validate.shape)

    return scores_train, params_train, scores_validate, params_validate


def plot_pc12(scores_train: np.ndarray, scores_validate: np.ndarray, show_plot=True):
    plt.figure(figsize=(10, 10))
    plt.scatter(scores_train[:, 0], scores_train[:, 1], label="Train")
    plt.scatter(scores_validate[:, 0], scores_validate[:, 1], label="Validate")
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "pc1_pc2.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_param_hists(params_train: np.ndarray, params_validate: np.ndarray, show_plot=True):
    plt.figure(figsize=(20, 10))
    for i in range(params_train.shape[1]):
        plt.subplot(3, 6, i + 1)
        plt.hist(params_train[:, i], bins=50, label="Train")
        plt.hist(params_validate[:, i], bins=50, label="Validate")
        if i == 0:
            plt.legend()

    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "parameters.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def fit_scalers(scores: np.ndarray, params: np.ndarray):
    score_scaler = MinMaxScaler(feature_range=(-1, 1), clip=True).fit(scores)
    params_scaler = StandardScaler().fit(params)

    return score_scaler, params_scaler


def plot_losses(losses, show_plot=True):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(losses["train"])
    plt.plot(losses["val"])
    plt.title("Total Loss")

    plt.subplot(132)
    plt.plot(losses["train_recon"])
    plt.plot(losses["val_recon"])
    plt.title("Reconstruction Loss")

    plt.subplot(133)
    plt.plot(losses["train_kl"])
    plt.plot(losses["val_kl"])
    plt.title("KL Loss")

    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "losses.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pc12_recon(
    scores_train,
    scores_train_recon,
    scores_validate,
    scores_validate_recon,
    show_plot=True,
):
    plt.scatter(
        scores_train[:, 0],
        scores_train[:, 1],
        c="gray",
        alpha=0.1,
        label="Train",
    )
    plt.scatter(
        scores_train_recon[:, 0],
        scores_train_recon[:, 1],
        label="Train Reconstruction",
    )
    plt.scatter(
        scores_validate_recon[:, 0],
        scores_validate_recon[:, 1],
        edgecolors="k",
        label="Validate Reconstruction",
    )
    plt.quiver(
        scores_validate[:, 0],
        scores_validate[:, 1],
        scores_validate_recon[:, 0] - scores_validate[:, 0],
        scores_validate_recon[:, 1] - scores_validate[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        headlength=5,
        headaxislength=4.5,
        minshaft=1,
        minlength=0.1,
        color="r",
        width=0.002,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "pc1_pc2_recon.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_recon_error_histograms(
    scores_train: np.ndarray,
    scores_train_recon: np.ndarray,
    scores_validate: np.ndarray,
    scores_validate_recon: np.ndarray,
    show_plot=True,
):
    train_recon_errors = np.mean(scores_train_recon - scores_train, axis=1)
    validate_recon_errors = np.mean(scores_validate_recon - scores_validate, axis=1)

    plt.figure(figsize=(10, 5))
    plt.hist(train_recon_errors, bins=100, density=True, alpha=0.5, label="Train")
    plt.hist(validate_recon_errors, bins=100, density=True, alpha=0.5, label="Validate")
    plt.xlabel("Reconstruction Error (MAE) Average Across PCs")
    plt.ylabel("Density")
    plt.legend()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "reconstruction_error_histogram.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_parity_plots(
    scores_train,
    scores_train_recon,
    scores_validate,
    scores_validate_recon,
    show_plot=True,
):
    nx = 4  # columns
    ny = 4  # rows

    pcs_to_plot = nx * ny

    plt.figure(figsize=(4 * nx, 4 * ny))  # dynamic figure size

    for i in range(pcs_to_plot):
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
            label="Validate",
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
        plt.title(f"PC{i + 1} Parity Plot")
        plt.legend()

    plt.suptitle(f"Parity Plots for First {pcs_to_plot} PC Scores")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "parity_plots.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_relative_MAE_vs_score(
    scores_train: np.ndarray,
    scores_train_recon: np.ndarray,
    scores_validate: np.ndarray,
    scores_validate_recon: np.ndarray,
    show_plot=True,
):
    # Compute range (max - min) per PC from training set
    pc_min = np.min(scores_train, axis=0)
    pc_max = np.max(scores_train, axis=0)
    pc_range = pc_max - pc_min

    # Calculate Relative MAE per PC (as a fraction of range)
    rel_mae_train = []
    rel_mae_val = []

    n_pcs = scores_train.shape[-1]

    for i in range(n_pcs):
        mae_train_i = mean_absolute_error(scores_train[:, i], scores_train_recon[:, i])
        mae_val_i = mean_absolute_error(
            scores_validate[:, i], scores_validate_recon[:, i]
        )

        rel_mae_train.append(
            mae_train_i / (pc_range[i] + 1e-8)
        )  # avoid division by zero
        rel_mae_val.append(mae_val_i / (pc_range[i] + 1e-8))

    rel_mae_train = np.array(rel_mae_train)
    rel_mae_val = np.array(rel_mae_val)

    # Convert to %
    rel_mae_train_pct = 100 * rel_mae_train
    rel_mae_val_pct = 100 * rel_mae_val

    # Moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    window = 3
    x = np.arange(1, n_pcs + 1)
    ma_x = np.arange(window // 2 + 1, n_pcs - window // 2 + 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        x,
        rel_mae_train_pct,
        label="Train Relative MAE (%)",
        color="tab:blue",
        alpha=0.6,
    )
    plt.scatter(
        x,
        rel_mae_val_pct,
        label="Validate Relative MAE (%)",
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
        moving_average(rel_mae_val_pct, window),
        color="tab:orange",
        linewidth=2,
        label="Validate Moving Avg",
    )

    plt.xlabel("PC Index")
    plt.ylabel("Relative MAE (% of PC Range)")
    plt.title("Relative Reconstruction MAE per PC with Moving Average")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "relative_mae_vs_pc_scores.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return rel_mae_train, rel_mae_val


def plot_2ps_examples(stats: np.ndarray, title: str, show_plot=True):
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(stats[i], cmap="inferno")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Sample {i}")
        plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / f"{'_'.join(title.lower().split(' '))}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_2ps_recon_examples(
    stats: np.ndarray,
    stats_recon: np.ndarray,
    title: str,
    show_plot=True,
):
    plt.figure(figsize=(15, 10))
    for i in range(9):
        stats_og = stats[i]
        stats_re = stats_recon[i]

        vmin = np.min([stats_og, stats_re])
        vmax = np.max([stats_og, stats_re])

        plt.subplot(3, 6, 2 * i + 1)
        plt.imshow(stats_og, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Original Sample {i}")

        plt.subplot(3, 6, 2 * i + 2)
        plt.imshow(stats_re, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Reconstructed Sample {i}")

    plt.suptitle(f"Original vs Reconstructed Stats : {title.title()}")
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.PATHS["figures_dir"]
        / f"{'_'.join(title.lower().split(' '))}_stats_recon_comparison.png"
    )

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_latent_histograms(z_train, z_val, show_plot=True):
    # Plot
    num_dims = min(z_train.shape[0], 16)
    cols = 4
    rows = int(np.ceil(num_dims / cols))

    num_bins = 40

    plt.figure(figsize=(4 * cols, 3 * rows))

    for i in range(num_dims):
        plt.subplot(rows, cols, i + 1)

        # Histogram: train
        plt.hist(
            z_train[:, i],
            bins=num_bins,
            density=True,
            alpha=0.5,
            label="Train",
            color="tab:blue",
        )

        # Histogram: val
        plt.hist(
            z_val[:, i],
            bins=num_bins,
            density=True,
            alpha=0.5,
            label="Validation",
            color="tab:orange",
        )

        # Standard normal
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x), "k--", label="Standard Normal")

        plt.title(f"z[{i}]")
        plt.xlabel("Value")
        plt.ylabel("Density")
        if i == 0:
            plt.legend()

    plt.suptitle("Latent Space Histograms : Train vs Validate vs Standard Normal")
    plt.tight_layout()

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "latent_histograms_KL_check.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_KL_vs_latent_dim(z_mu_train, z_logvar_train, z_mu_val, z_logvar_val):
    """
    Compute the true analytical KL divergence per latent dimension for both
    training and validation sets.

    Returns
    -------
    kl_train : np.ndarray
        Average KL divergence per latent dim for training set.
    kl_val : np.ndarray
        Average KL divergence per latent dim for validation set.
    """
    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu**2 - np.exp(logvar))  # shape: [N, D]

    kl_train = kl_div(z_mu_train, z_logvar_train).mean(axis=0)
    kl_val = kl_div(z_mu_val, z_logvar_val).mean(axis=0)

    return kl_train, kl_val


def plot_KL_vs_latent_dim(kl_train, kl_val, show_plot=True):
    x = np.arange(1, len(kl_train) + 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, kl_train, color="tab:blue", label="Train KL Divergence")
    plt.scatter(x, kl_val, color="tab:orange", label="Validation KL Divergence")
    plt.plot(x, kl_train, color="tab:blue", alpha=0.5)
    plt.plot(x, kl_val, color="tab:orange", alpha=0.5)

    avg_kl_train = kl_train.mean()
    avg_kl_val = kl_val.mean()
    plt.axhline(
        avg_kl_train,
        color="tab:blue",
        linestyle="--",
        linewidth=3,
        label=f"Train Avg: {avg_kl_train:.3f}",
    )
    plt.axhline(
        avg_kl_val,
        color="tab:orange",
        linestyle="--",
        linewidth=3,
        label=f"Val Avg: {avg_kl_val:.3f}",
    )

    plt.xlabel("Latent Dimension")
    plt.ylabel("KL Divergence to Standard Normal")
    plt.title("True KL Divergence per Latent Dimension (Lower is Better)")
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "kl_div_vs_latent_dim.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_vae_sample_histograms(
    scores_train,
    scores_validate,
    vae_samples,
    validation_samples,
    show_plot=True,
):
    """
    Plot histograms of first 8 PCs of train+val scores, VAE samples for each validation sample, and a red vline for the validation sample.

    Parameters
    ----------
    scores_train : np.ndarray
        PC scores for training set
    scores_validate : np.ndarray
        PC scores for validation set
    vae_samples : np.ndarray
        VAE samples for each validation sample (n_val, n_vae, n_scores)
    validation_samples : np.ndarray
        Validation samples (n_val, n_scores)

    Returns
    -------
    None
    """

    # Histogram for first 8 PCs
    num_pcs = 8
    # Gather all PC scores from train+val for histograms
    all_scores = np.vstack([scores_train, scores_validate])

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

    for i_val in range(validation_samples.shape[0]):
        plt.figure(figsize=(20, 8))
        for pc in range(num_pcs):
            plt.subplot(2, 4, pc + 1)

            # Histogram of all train+val for this PC
            plt.hist(
                all_scores[:, pc],
                bins=40,
                density=True,
                color="gray",
                alpha=0.7,
                label="PC Distribution",
            )

            # Blue hsit for samples
            plt.hist(
                vae_samples[i_val, :, pc],
                bins=40,
                density=True,
                color="blue",
                alpha=0.7,
                label="VAE Samples",
            )

            # Red vline for the validation sample
            plt.axvline(
                x=validation_samples[i_val, pc],
                color="red",
                lw=2,
                label="Validation Sample",
            )

            plt.ylabel("Density")
            plt.xlabel(f"PC{pc + 1}")
            plt.title(f"Samples in PC{pc + 1}")
            if pc == 0:
                plt.legend()

        plt.suptitle(
            f"Val sample {i_val + 1}: VAE samples (blue) vs true (red) in PC space"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fig = plt.gcf()
        fig.savefig(
            config.PATHS["figures_dir"] / f"val_sample_{i_val + 1}_vae_samples_scores.png"
        )

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_vae_sample_reconstructions(stats_val, stats_vae, show_plot=True):
    # Compare a few sample reconstructions to the original validation reconstruction (stats/image)
    # Inverse transform from PCs to stats/images for both val and VAE samples

    n_val = stats_val.shape[0]
    n_show = stats_vae.shape[1]

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

    for i_val in range(n_val):
        vmin = np.min([stats_val[i_val]] + [stats_vae[i_val, j] for j in range(n_show)])
        vmax = np.max([stats_val[i_val]] + [stats_vae[i_val, j] for j in range(n_show)])

        plt.figure(figsize=(3 * (n_show + 1), 3))
        # First column: the true validation sample
        plt.subplot(1, n_show + 1, 1)
        plt.imshow(stats_val[i_val], cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title("Validation\nSample")
        plt.colorbar()
        plt.axis("off")
        # Next columns: the VAE reconstructions
        for j in range(n_show):
            plt.subplot(1, n_show + 1, j + 2)
            plt.imshow(stats_vae[i_val, j], cmap="inferno", vmin=vmin, vmax=vmax)
            plt.title(f"VAE\nSample {j + 1}")
            plt.axis("off")
        plt.suptitle(f"Validation sample {i_val + 1}: true (left) vs VAE samples")
        plt.tight_layout()

        fig = plt.gcf()
        fig.savefig(
            config.PATHS["figures_dir"] / f"val_sample_{i_val + 1}_vae_samples_stats.png"
        )

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_vae_sample_cross_sections(stats_val_img, stats_vae_img, show_plot=True):
    n_val, n_vae, im_size, _ = (
        stats_vae_img.shape
    )  # stats_vae_img should now be (n_val, n_vae, im_h, im_w)
    center_row = im_size // 2

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

    for i in range(n_val):
        plt.figure(figsize=(12, 4))

        vae_cross_sections = stats_vae_img[i, :, center_row, :]

        # Plot all VAE sample cross-sections as gray lines
        for j in range(n_vae):
            plt.plot(
                vae_cross_sections[j],
                color="gray",
                alpha=0.1,
                linewidth=0.5,
            )

        # Plot mean VAE cross-section (red dashed)
        mean_cross_section = vae_cross_sections.mean(axis=0)
        plt.plot(
            mean_cross_section,
            label="VAE Mean",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Plot the original cross-section
        plt.plot(
            stats_val_img[i, center_row, :],
            label="Original",
            linewidth=2,
            color="black",
        )

        plt.title(f"Cross-Section View (Validation Sample {i + 1})")
        plt.xlabel("Pixel Index (Column)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig = plt.gcf()
        fig.savefig(config.PATHS["figures_dir"] / f"cross_section_val_{i+1}.png")

        if show_plot:
            plt.show()
        else:
            plt.close()


def decode_single_dim_traversal(
    model,
    score_scaler,
    pca,
    dim_idx: int,
    base_z: np.ndarray,
    base_cond_scaled: np.ndarray,
    latent_range: tuple[float, float] = (-3, 3),
    n_traverse: int = 100,
):
    """
    Decode a traversal of a single latent dimension, holding all other
    dimensions and the condition fixed, and return the resulting 2PS
    fields' central-row cross-sections.

    Parameters
    ----------
    base_z : (latent_dim,)
        Fixed base latent point (e.g. posterior mean of a representative
        test sample). Only `dim_idx` is varied; all other entries are held
        at their base_z value.
    base_cond_scaled : (cond_dim,)
        Fixed, already-scaled condition vector.

    Returns
    -------
    traverse_range : (n_traverse,)
    cross_sections : (n_traverse, im_size)
        Central-row cross-section of the decoded 2PS field at each
        traversal step.
    """
    latent_dim = base_z.shape[0]
    traverse_range = np.linspace(*latent_range, n_traverse, dtype=np.float32)

    Z = np.tile(base_z.astype(np.float32), (n_traverse, 1))  # (n_traverse, latent_dim)
    Z[:, dim_idx] = traverse_range

    C = np.tile(base_cond_scaled.astype(np.float32), (n_traverse, 1))  # (n_traverse, cond_dim)

    decoded_scaled = model.decode(z=Z, c=C)
    decoded_scores = score_scaler.inverse_transform(decoded_scaled)  # (n_traverse, n_scores)

    stats = pca.inverse_transform(decoded_scores)  # (n_traverse, n_pixels)
    im_size = int(stats.shape[-1] ** 0.5)
    stats_img = stats.reshape(n_traverse, im_size, im_size)

    center_row = im_size // 2
    cross_sections = stats_img[:, center_row, :]  # (n_traverse, im_size)

    return traverse_range, cross_sections


def plot_traversal_2ps_overlay(
    traverse_range: np.ndarray,
    cross_sections: np.ndarray,
    dim_idx: int,
    cmap: str = "plasma",
    title: str = None,
    show_plot: bool = True,
):
    """
    Plot 2PS central-row cross-sections for a single-dimension latent
    traversal, overlaid on one axes and colored by the traversal value.

    Parameters
    ----------
    traverse_range : (n_traverse,)
    cross_sections : (n_traverse, im_size)
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    norm_ = plt.Normalize(vmin=traverse_range.min(), vmax=traverse_range.max())
    cmap_ = plt.get_cmap(cmap)

    for i, v in enumerate(traverse_range):
        ax.plot(cross_sections[i], color=cmap_(norm_(v)), linewidth=0.8, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm_)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"$z_{{{dim_idx}}}$ value")

    ax.set_xlabel("Pixel Index (Column)")
    ax.set_ylabel("Intensity")
    ax.set_title(title if title is not None else f"2PS Cross-Section vs. Latent Dim {dim_idx} Traversal")

    plt.tight_layout()

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / f"latent_traversal_2ps_dim{dim_idx}.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_recon_pc_rug_plots(
    scores_train: np.ndarray,
    vae_samples_unscaled: np.ndarray,
    recon_samples_unscaled: np.ndarray,
    show_plot: bool = True,
    n_pcs: int = 8,
):
    n_samples = vae_samples_unscaled.shape[0]

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        fig, axs = plt.subplots(2, 4, figsize=(15, 6), sharey=False)
        axs = axs.flatten()
        fig.suptitle(
            f"PC Distribution: VAE Samples vs Reconstructions - Sample {i + 1}",
            fontsize=16,
        )

        for j in range(n_pcs):
            ax = axs[j]
            ax.vlines(
                scores_train[:, j],
                ymin=0.0, ymax=0.3,
                color="lightgray", alpha=0.2, linewidth=1.5,
                label="PC Distribution" if j == 0 else None,
            )
            ax.vlines(
                vae_samples_unscaled[i, :, j],
                ymin=0.3, ymax=0.6,
                color="tab:orange", alpha=0.25, linewidth=1.5,
                label="VAE Samples" if j == 0 else None,
            )
            ax.vlines(
                recon_samples_unscaled[i, :, j],
                ymin=0.6, ymax=0.9,
                color="tab:blue", alpha=0.25, linewidth=1.5,
                label="Reconstructions" if j == 0 else None,
            )

            ax.set_title(f"PC {j + 1}")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_yticks([])

        # Build opaque proxy handles so legend colors are vivid
        legend_handles = [
            Line2D([0], [0], color="lightgray", lw=4, alpha=1, label="PC Distribution"),
            Line2D([0], [0], color="tab:orange", lw=4, alpha=1, label="VAE Samples"),
            Line2D([0], [0], color="tab:blue", lw=4, alpha=1, label="Reconstructions"),
        ]
        axs[0].legend(handles=legend_handles, loc="upper left")

        plt.tight_layout()

        fig.savefig(config.PATHS["figures_dir"] / f"recon_pc_rug_plots_{i + 1}.png")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_pc_vs_composition_trend_with_variance(
    comp_values: np.ndarray,
    mean_pc: np.ndarray,
    std_pc: np.ndarray,
    show_plot: bool = True,
    n_pcs: int = 8,
):
    """
    Plots mean ± 1 std of the first n_pcs PCs across a composition sweep.

    Parameters
    ----------
    comp_values : (n_comp,)
    mean_pc : (n_comp, n_scores)
    std_pc : (n_comp, n_scores)
    """
    fig, axs = plt.subplots(2, 4, figsize=(15, 6), sharey=False)
    axs = axs.flatten()

    for i in range(n_pcs):
        mean_vals = mean_pc[:, i]
        std_vals = std_pc[:, i]

        axs[i].plot(comp_values, mean_vals, label="Mean")
        axs[i].fill_between(
            comp_values,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.3,
            label="±1 STD" if i == 0 else None,
        )
        axs[i].set_title(f"PC {i + 1} vs Composition")
        axs[i].set_xlabel("Composition")
        axs[i].set_ylabel("PC Value")

    axs[0].legend()
    plt.suptitle("Conditional Trends with Variance: Composition Sweep", fontsize=16)
    plt.tight_layout()

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "pc_vs_composition_trend_with_variance.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_latent_diagonal_pc_trend(
    traverse_range: np.ndarray,
    decoded_scores: np.ndarray,
    show_plot: bool = True,
    n_pcs: int = 8,
):
    """
    Plots first n_pcs PCs versus a deterministic diagonal traversal of latent space.

    Parameters
    ----------
    traverse_range : (n_traverse,)
    decoded_scores : (n_traverse, n_scores)
    """
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    axs = axs.flatten()

    for i in range(n_pcs):
        axs[i].plot(
            traverse_range, decoded_scores[:, i], color="tab:green", label="PC Value"
        )
        axs[i].set_title(f"PC {i + 1} vs Latent Diagonal")
        axs[i].set_xlabel("z = [-3 ... +3] (all dims)")
        axs[i].set_ylabel("PC Value")

    plt.suptitle("Latent Diagonal Traversal", fontsize=16)
    plt.tight_layout(h_pad=3.0)

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "latent_diagonal_pc_trend_deterministic.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_distribution_fidelity_pc_histograms(
    reference_pc: np.ndarray,
    sampled_pc: np.ndarray,
    show_plot: bool = True,
    label_reference: str = "Validate",
    label_samples: str = "CVAE Samples",
    n_pcs: int = 8,
    bins: int = 50,
):
    """
    Overlaid histograms comparing marginal PC distributions between
    a reference set and CVAE-sampled set.

    Parameters
    ----------
    reference_pc : (N_ref, n_scores)
    sampled_pc : (N_sampled, n_scores)
    """
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    axs = axs.flatten()

    for i in range(n_pcs):
        axs[i].hist(
            reference_pc[:, i],
            bins=bins,
            alpha=0.5,
            color="gray",
            label=label_reference,
            density=True,
        )
        axs[i].hist(
            sampled_pc[:, i],
            bins=bins,
            alpha=0.5,
            color="tab:blue",
            label=label_samples,
            density=True,
        )
        axs[i].set_title(f"PC {i + 1}")
        axs[i].tick_params(axis="x", labelrotation=45)

    axs[0].legend()
    plt.tight_layout(h_pad=2.0)

    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PATHS["figures_dir"] / "distribution_fidelity_pc_histograms.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
