"""
Epistemic vs. Aleatoric Uncertainty Analysis (Fig. 11)

Produces cross-section plots comparing sampling variability (aleatoric) and
model variability (epistemic) for 5 selected test samples, using the
canonical VAE (config.PATHS["vae"]) plus the ensemble members produced by
scripts/04_epistemic_ensemble.py (config.DATA_DIR / "vae_ensemble_run_{i}.pkl").

Ported from code/_epistemic_uncertainty.py.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src_plotting._final_vae_utils import pick_example_indices

import config

######## CONFIG ########

N_SCORES = config.N_SCORES
N_VAE_CS = 1000
N_EPI_SAMPLES = 500
N_ENSEMBLE_RUNS = 10  # matches scripts/04_epistemic_ensemble.py N_RUNS


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca


if __name__ == "__main__":
    ######## LOAD SHARED DATA ########

    with open(config.PATHS["scaler_scores"], "rb") as f:
        scaler_scores = pickle.load(f)

    with open(config.PATHS["scaler_params"], "rb") as f:
        scaler_params = pickle.load(f)

    with open(config.PATHS["pca"], "rb") as f:
        pca: PCA = pickle.load(f)
        pca = trim_pca(pca, N_SCORES)

    with h5py.File(config.PATHS["segmented_data"], "r") as f:
        params_test = f["hetero_only"]["test"]["params"][...]
        stats_test = f["hetero_only"]["test"]["stats"][..., 0]  # (n, H, W)

    ######## SELECT SAMPLES (matches Figure 12) ########

    test_stats_mmae_path = config.DATA_DIR / "metrics__analytics__mmae_test_stats_sampling_error.npy"
    test_stats_mmae = np.load(test_stats_mmae_path)
    selected_inds = pick_example_indices(test_stats_mmae)  # [11, 53, 14, 12, 148]
    selected_points = params_test[selected_inds]

    stats_test_selected = stats_test[selected_inds]
    im_size = stats_test_selected.shape[1]
    center_row = im_size // 2

    ######## SAMPLE: ALEATORIC (canonical model) ########

    with open(config.PATHS["vae"], "rb") as f:
        canonical_model = pickle.load(f)

    aleatoric_vae_imgs = np.zeros((len(selected_inds), N_VAE_CS, im_size, im_size))
    for rep_idx, base_param in enumerate(selected_points):
        c = scaler_params.transform(base_param[np.newaxis, :])
        c_repeat = np.repeat(c, N_VAE_CS, axis=0)
        samples_scaled = canonical_model.sample(num_samples=N_VAE_CS, c=c_repeat)
        samples_unscaled = scaler_scores.inverse_transform(samples_scaled)
        aleatoric_vae_imgs[rep_idx] = pca.inverse_transform(samples_unscaled).reshape(
            N_VAE_CS, im_size, im_size
        )

    print("Aleatoric sampling complete.")

    ######## SAMPLE: EPISTEMIC (all ensemble models) ########

    model_paths = [config.PATHS["vae"]] + [
        config.DATA_DIR / f"vae_ensemble_run_{i}.pkl" for i in range(N_ENSEMBLE_RUNS)
    ]
    n_runs = len(model_paths)

    epistemic_model_means = np.zeros((n_runs, len(selected_inds), im_size))

    for run_idx, model_path in enumerate(model_paths):
        print(f"Epistemic model {run_idx + 1}/{n_runs}: {model_path}")
        with open(model_path, "rb") as f:
            epi_model = pickle.load(f)

        for rep_idx, base_param in enumerate(selected_points):
            c = scaler_params.transform(base_param[np.newaxis, :])
            c_repeat = np.repeat(c, N_EPI_SAMPLES, axis=0)
            samples_scaled = epi_model.sample(num_samples=N_EPI_SAMPLES, c=c_repeat)
            samples_unscaled = scaler_scores.inverse_transform(samples_scaled)
            imgs = pca.inverse_transform(samples_unscaled).reshape(
                N_EPI_SAMPLES, im_size, im_size
            )
            epistemic_model_means[run_idx, rep_idx] = imgs[:, center_row, :].mean(axis=0)

    epistemic_model_means = epistemic_model_means.transpose(1, 0, 2)  # -> (n_samples, n_runs, im_size)

    print("Epistemic sampling complete.")

    ######## PLOT ########

    def plot_epistemic_aleatoric_cross_sections(
        stats_test_img,
        aleatoric_vae_imgs,
        epistemic_model_means,
        sample_labels=None,
        show_plot=True,
    ):
        if sample_labels is None:
            sample_labels = [
                "Min Error Sample",
                "Near Mean Error Sample",
                "Near Mean Error Sample",
                "Near Mean Error Sample",
                "Max Error Sample",
            ]

        n_test, n_vae, im_size, _ = aleatoric_vae_imgs.shape
        center_row = im_size // 2

        config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

        for i in range(n_test):
            vae_cross_sections = aleatoric_vae_imgs[i, :, center_row, :]
            epi_means = epistemic_model_means[i]
            orig = stats_test_img[i, center_row, :].squeeze()

            ale_std = vae_cross_sections.std(axis=0)
            epi_std = epi_means.std(axis=0)

            fig, (ax, ax_std) = plt.subplots(1, 2, figsize=(12, 2.25))

            for j in range(n_vae):
                ax.plot(vae_cross_sections[j], color="gray", alpha=0.1, linewidth=0.5)

            mean_cross_section = vae_cross_sections.mean(axis=0)
            ind_mean = np.argmin(np.abs(vae_cross_sections - mean_cross_section).sum(axis=1))
            ax.plot(vae_cross_sections[ind_mean], label="Mean Sample", color="red",
                    linestyle="--", linewidth=2)

            ind_closest = np.argmin(np.abs(vae_cross_sections - orig).sum(axis=1))
            ax.plot(vae_cross_sections[ind_closest], label="Closest Sample", color="blue",
                    linestyle="--", linewidth=2)

            ax.plot(orig, label="Original", linewidth=2, color="black")

            data_min = np.min(vae_cross_sections)
            data_max = np.max(vae_cross_sections)
            pad = 0.05 * (data_max - data_min)
            ax.set_ylim(data_min - pad, data_max + pad)

            ax.set_xlabel("Pixel Index in Middle Cross-Section", fontsize=11)
            ax.set_ylabel("Correlation", fontsize=11)
            ax.tick_params(labelsize=12)
            ax.grid(True)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[k] for k in [2, 0, 1]], [labels[k] for k in [2, 0, 1]],
                      fontsize=11, loc="upper right")

            x = np.arange(im_size)
            ax_std.plot(x, ale_std, color="gray", linewidth=1.5, label="Sampling Variability")
            ax_std.plot(x, epi_std, color="blue", linewidth=1.5, label="Model Variability")
            ax_std.set_xlabel("Pixel Index in Middle Cross-Section", fontsize=11)
            ax_std.set_ylabel("Standard Deviation", fontsize=11)
            ax_std.tick_params(labelsize=12)
            ax_std.legend(fontsize=11, loc="upper right")
            ax_std.grid(True)

            fig.tight_layout()

            path = config.PATHS["figures_dir"] / f"cross_section_epi_ale_{i + 1}.png"
            fig.savefig(path, dpi=300)
            print(f"Saved {path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

    plot_epistemic_aleatoric_cross_sections(
        stats_test_img=stats_test_selected,
        aleatoric_vae_imgs=aleatoric_vae_imgs,
        epistemic_model_means=epistemic_model_means,
        sample_labels=[
            "Min Error",
            "Near Mean Error",
            "Near Mean Error",
            "Near Mean Error",
            "Max Error",
        ],
        show_plot=True,
    )
