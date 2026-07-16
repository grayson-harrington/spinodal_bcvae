from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# %%

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from spinodal_cvae import VariationalAutoencoder

from src_plotting.vae_utils import (
    load_hetero_only_data,
    plot_pc12,
    plot_param_hists,
    fit_scalers,
    plot_losses,
    plot_pc12_recon,
    plot_recon_error_histograms,
    plot_parity_plots,
    plot_relative_MAE_vs_score,
    plot_2ps_examples,
    plot_2ps_recon_examples,
    plot_latent_histograms,
    get_KL_vs_latent_dim,
    plot_KL_vs_latent_dim,
    plot_vae_sample_histograms,
    plot_vae_sample_reconstructions,
    plot_vae_sample_cross_sections,
)

import optuna

import config


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca


n_trials = 300


def suggest_encoder_shape(trial: optuna.Trial, max_layers=2, max_size=512):
    n_layers = trial.suggest_int("n_layers", 1, max_layers)
    sizes = []
    prev_size = max_size
    for i in range(n_layers):
        size = trial.suggest_int(f"layer_{i}_size", 64, prev_size, step=64)
        sizes.append(size)
        prev_size = size
    return tuple(sizes)


def make_objective(
    scores_train,
    params_train,
    scores_validate,
    params_validate,
    score_scaler,
    params_scaler,
    scores_train_scaled,
    scores_validate_scaled,
    params_train_scaled,
    params_validate_scaled,
):
    def objective(trial: optuna.Trial):
        plot_pc12(scores_train, scores_validate, show_plot=False)
        plot_param_hists(params_train, params_validate, show_plot=False)

        ######## MODEL TRAINING ########

        n_epochs = 300
        n_epochs_cosine_annealing = 300

        model = VariationalAutoencoder(
            # Architecture hyperparameters
            encoder_shape=suggest_encoder_shape(trial),
            latent_dim=32,
            activation_function="gelu",
            # Beta-VAE hyperparameters
            beta=trial.suggest_float("beta", 1e-6, 1e-2, log=True),
            kl_warmup_epochs=trial.suggest_int("kl_warmup_epochs", 10, 50, step=5),
            kl_annealing_epochs=trial.suggest_int("kl_annealing_epochs", 25, 100, step=5),
            free_bits_threshold=trial.suggest_float(
                "free_bits_threshold", 0, 0.2, step=0.05
            ),
            # Training hyperparameters
            optimizer="adam",
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            scheduler="cosine_annealing",
            T_max=n_epochs_cosine_annealing,
            n_epochs=n_epochs,
            batch_size=64,
        )

        print(f"hyperparameters: {model.hyperparameters}")

        model.fit(
            X_train=scores_train_scaled,
            y_train=scores_train_scaled,
            c_train=params_train_scaled,
            X_validation=scores_validate_scaled,
            y_validation=scores_validate_scaled,
            c_validation=params_validate_scaled,
        )

        losses = model.losses

        plot_losses(losses, show_plot=False)

        scores_train_recon_scaled = model.predict(scores_train_scaled, params_train_scaled)
        scores_validate_recon_scaled = model.predict(
            scores_validate_scaled, params_validate_scaled
        )

        scores_train_recon = score_scaler.inverse_transform(scores_train_recon_scaled)
        scores_validate_recon = score_scaler.inverse_transform(scores_validate_recon_scaled)

        train_recon_errors = np.mean(scores_train_recon - scores_train_scaled, axis=1)
        validate_recon_errors = np.mean(
            scores_validate_recon - scores_validate_scaled, axis=1
        )

        # Check for NaNs in reconstruction errors
        if np.any(np.isnan(train_recon_errors)) or np.any(np.isnan(validate_recon_errors)):
            print("NaNs detected in reconstruction errors — pruning trial.")
            raise optuna.exceptions.TrialPruned()

        plot_pc12_recon(
            scores_train,
            scores_train_recon,
            scores_validate,
            scores_validate_recon,
            show_plot=False,
        )

        plot_recon_error_histograms(
            scores_train,
            scores_train_recon,
            scores_validate,
            scores_validate_recon,
            show_plot=False,
        )

        plot_parity_plots(
            scores_train,
            scores_train_recon,
            scores_validate,
            scores_validate_recon,
            show_plot=False,
        )

        rel_mae_train, rel_mae_val = plot_relative_MAE_vs_score(
            scores_train,
            scores_train_recon,
            scores_validate,
            scores_validate_recon,
            show_plot=False,
        )

        average_rel_mae_train = float(np.mean(rel_mae_train))
        average_rel_mae_val = float(np.mean(rel_mae_val))
        print(f"average_rel_mae_train: {average_rel_mae_train}")
        print(f"average_rel_mae_val: {average_rel_mae_val}")

        # test sampling from the VAE and reconstructing
        #   get hundreds of samples for different initial conditions and look at distrubtions for the PCs
        #   do for real samples and see if the reconstructions look appropriate or not..

        with open(config.PATHS["pca"], "rb") as f:
            pca: PCA = pickle.load(f)
            pca = trim_pca(pca, config.N_SCORES)

        # get stuff for checking train
        scores_train_check = scores_train
        params_train_check = params_train

        stats_train_check = pca.inverse_transform(scores_train_check)
        stats_train_check = stats_train_check.reshape(
            -1, *tuple([int(stats_train_check.shape[-1] ** 0.5)] * 2)
        )

        # get stuff for checking validation
        scores_validate_check = scores_validate
        params_validate_check = params_validate

        stats_validate_check = pca.inverse_transform(scores_validate_check)
        stats_validate_check = stats_validate_check.reshape(
            -1, *tuple([int(stats_validate_check.shape[-1] ** 0.5)] * 2)
        )

        plot_2ps_examples(stats_train_check, "Train Examples", show_plot=False)
        plot_2ps_examples(stats_validate_check, "Validate Examples", show_plot=False)

        # get stuff for checking training
        scores_train_check = score_scaler.transform(scores_train_check)
        params_train_check = params_scaler.transform(params_train_check)

        scores_train_recon_check = model.predict(X=scores_train_check, c=params_train_check)
        scores_train_recon_check = score_scaler.inverse_transform(scores_train_recon_check)

        stats_train_recon_check = pca.inverse_transform(scores_train_recon_check)
        stats_train_recon_check = stats_train_recon_check.reshape(
            -1, *tuple([int(stats_train_recon_check.shape[-1] ** 0.5)] * 2)
        )

        # get stuff for checking validation
        scores_validate_check = score_scaler.transform(scores_validate_check)
        params_validate_check = params_scaler.transform(params_validate_check)

        scores_validate_recon_check = model.predict(
            X=scores_validate_check, c=params_validate_check
        )
        scores_validate_recon_check = score_scaler.inverse_transform(
            scores_validate_recon_check
        )

        stats_validate_recon_check = pca.inverse_transform(scores_validate_recon_check)
        stats_validate_recon_check = stats_validate_recon_check.reshape(
            -1, *tuple([int(stats_validate_recon_check.shape[-1] ** 0.5)] * 2)
        )

        plot_2ps_recon_examples(
            stats_train_check,
            stats_train_recon_check,
            "Training Set",
            show_plot=False,
        )

        plot_2ps_recon_examples(
            stats_validate_check,
            stats_validate_recon_check,
            "Validation Set",
            show_plot=False,
        )

        # Encode latent means
        z_mu_train, z_logvar_train = model.encode(
            X=scores_train_check, c=params_train_check
        )
        z_mu_val, z_logvar_val = model.encode(
            X=scores_validate_check, c=params_validate_check
        )

        z_std_train = np.exp(0.5 * z_logvar_train)
        z_std_val = np.exp(0.5 * z_logvar_val)

        z_train = z_mu_train + z_std_train * np.random.randn(*z_mu_train.shape)
        z_val = z_mu_val + z_std_val * np.random.randn(*z_mu_val.shape)

        plot_latent_histograms(z_train, z_val, show_plot=False)

        # Plot KL vs latent dim
        kl_train, kl_val = get_KL_vs_latent_dim(
            z_mu_train, z_logvar_train, z_mu_val, z_logvar_val
        )
        plot_KL_vs_latent_dim(kl_train, kl_val, show_plot=False)

        avg_kl_train = float(np.mean(kl_train))
        avg_kl_val = float(np.mean(kl_val))
        print(f"avg_kl_train: {avg_kl_train}")
        print(f"avg_kl_val: {avg_kl_val}")

        # Select 5 validation samples
        n_val = 5
        np.random.seed(42)
        inds_val = np.random.choice(len(scores_validate), size=n_val, replace=False)
        scores_val_selected = scores_validate[inds_val]
        params_val_selected = params_validate_scaled[inds_val]

        # Get a lot of VAE samples for each validation sample's params
        n_vae = 250
        vae_samples_scaled = []

        for c in params_val_selected:
            # Repeat each c n_vae times for sampling
            c_repeat = np.repeat(c[np.newaxis, :], n_vae, axis=0)
            vae_sampled_scores = model.sample(num_samples=n_vae, c=c_repeat)
            vae_samples_scaled.append(vae_sampled_scores)

        vae_samples_scaled = np.stack(vae_samples_scaled)  # shape (n_val, n_vae, n_scores)

        # Reconstruct to PC space using score_scaler
        vae_samples_unscaled = score_scaler.inverse_transform(
            vae_samples_scaled.reshape(-1, vae_samples_scaled.shape[-1])
        )
        vae_samples_unscaled = vae_samples_unscaled.reshape(n_val, n_vae, -1)

        plot_vae_sample_histograms(
            scores_train,
            scores_validate,
            vae_samples_unscaled,
            scores_val_selected,
            show_plot=False,
        )

        # Compare the 6 sample reconstructions to the original validation reconstruction (stats/image)
        # Inverse transform from PCs to stats/images for both val and VAE samples
        n_show = 5
        stats_val = pca.inverse_transform(scores_val_selected)
        s = vae_samples_unscaled.shape
        stats_vae = pca.inverse_transform(
            vae_samples_unscaled.reshape(-1, vae_samples_unscaled.shape[-1])
        )
        stats_vae = stats_vae.reshape(s[0], s[1], -1)

        # Reshape if needed for image display (assumes square images)
        im_size = int(stats_val.shape[-1] ** 0.5)
        stats_val_img = stats_val.reshape(n_val, im_size, im_size)
        stats_vae_img = stats_vae.reshape(n_val, s[1], im_size, im_size)

        plot_vae_sample_reconstructions(
            stats_val_img,
            stats_vae_img[:, :n_show],
            show_plot=False,
        )
        plot_vae_sample_cross_sections(
            stats_val_img,
            stats_vae_img,
            show_plot=False,
        )

        with open(config.DATA_DIR / f"vae_trial_{trial.number}.pkl", "wb") as f:
            pickle.dump(model, f)

        return average_rel_mae_val, avg_kl_val

    return objective


if __name__ == "__main__":
    scores_train, params_train, scores_validate, params_validate = load_hetero_only_data(
        config.PATHS["segmented_data"], config.N_SCORES
    )

    score_scaler, params_scaler = fit_scalers(scores_train, params_train)

    scores_train_scaled = score_scaler.transform(scores_train)
    scores_validate_scaled = score_scaler.transform(scores_validate)

    params_train_scaled = params_scaler.transform(params_train)
    params_validate_scaled = params_scaler.transform(params_validate)

    objective = make_objective(
        scores_train,
        params_train,
        scores_validate,
        params_validate,
        score_scaler,
        params_scaler,
        scores_train_scaled,
        scores_validate_scaled,
        params_train_scaled,
        params_validate_scaled,
    )

    # Multi-objective! Persisted via optuna's own SQLite storage so the study
    # (and any completed trials) survive across interrupted runs.
    study_name = "segmented_scores_vae_optimization"
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_trials // 5)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        storage=f"sqlite:///{config.DATA_DIR}/optuna_study.db",
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    # %%

    df: pd.DataFrame = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    df = df.drop(
        columns=[
            "number",
            "datetime_start",
            "datetime_complete",
            "duration",
            "state",
        ]
    )
    df.columns = [
        col.split("params_")[1] if col.startswith("params_") else col for col in df.columns
    ]
    df = df.rename(columns={"values_0": "val_recon_MAE", "values_1": "val_KL"})

    df = df[df["val_recon_MAE"] < 0.1]

    df.to_csv(config.DATA_DIR / "optimization_results.csv", index=False)

    print(df.head())

    # %%

    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.scatter(df["val_recon_MAE"], df["val_KL"], c=df["lr"], s=5, norm="log")
    plt.colorbar(label="learning rate")
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("KL Divergence from Unit Gaussian")
    plt.title("Pareto Front vs Learning Rate")

    plt.subplot(223)
    plt.scatter(df["val_recon_MAE"], df["val_KL"], c=df["layer_0_size"], s=5)
    plt.colorbar(label="first layer size")
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("KL Divergence from Unit Gaussian")
    plt.title("Pareto Front vs First Layer Size")

    plt.subplot(222)
    plt.scatter(df["val_recon_MAE"], df["val_KL"], c=df["beta"], s=5, norm="log")
    plt.colorbar(label="$\\beta$")
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("KL Divergence from Unit Gaussian")
    plt.title("Pareto Front vs $\\beta$")

    plt.subplot(224)
    plt.scatter(df["val_recon_MAE"], df["val_KL"], c=df["free_bits_threshold"], s=5)
    plt.colorbar(label="free bits threshold")
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("KL Divergence from Unit Gaussian")
    plt.title("Pareto Front vs Free Bits Threshold")

    plt.suptitle("Optimization Results", fontsize=24)
    plt.tight_layout(pad=2.0, h_pad=5.0, w_pad=5.0)
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    plt.savefig(config.PATHS["figures_dir"] / "optimization_results_vs_hyperparameters.png")
    plt.show()

    # %%

    plt.scatter(
        df["val_recon_MAE"],
        df["val_KL"],
        c=df["beta"],
        s=5,
        cmap="viridis",
        norm="log",
    )
    plt.colorbar(label="$\\beta$")
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("KL Divergence from Unit Gaussian")

    plt.title("Optimization Results")
    plt.savefig(config.PATHS["figures_dir"] / "optimization_results.png")
    plt.show()

    # %%

    import matplotlib as mpl

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
            "axes.labelsize": 10,
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

    # %%

    # Normalize both metrics
    recon_mae_norm = (df["val_recon_MAE"] - df["val_recon_MAE"].min()) / (
        df["val_recon_MAE"].max() - df["val_recon_MAE"].min()
    )
    kl_norm = (df["val_KL"] - df["val_KL"].min()) / (
        df["val_KL"].max() - df["val_KL"].min()
    )

    # Define utopia point (0, 0) in normalized space
    # Compute distance from each point to the utopia point
    dist_to_utopia = np.sqrt(recon_mae_norm**2 + kl_norm**2)

    # Find the index of the best point (closest to utopia)
    best_index = dist_to_utopia.idxmin()

    # Plot the scatter with original (non-normalized) values
    plt.figure(figsize=(5, 4))

    sc = plt.scatter(
        df["val_recon_MAE"],
        df["val_KL"],
        c=df["beta"],
        s=5,
        cmap="viridis",
        norm="log",
    )

    plt.colorbar(label="$\\beta$")

    # Plot the utopia point (min MAE, min KL)
    utopia_mae = df["val_recon_MAE"].min()
    utopia_kl = df["val_KL"].min()
    plt.axhline(utopia_kl, color="gray", linestyle="--", linewidth=1)
    plt.axvline(utopia_mae, color="gray", linestyle="--", linewidth=1)

    # Plot utopia point
    plt.scatter(
        utopia_mae,
        utopia_kl,
        color="blue",
        marker="o",
        s=10,
        label="Utopia Point",
        edgecolor="black",
    )

    # Plot the best trial (closest to utopia) with a red star
    plt.scatter(
        df.loc[best_index, "val_recon_MAE"],
        df.loc[best_index, "val_KL"],
        color="red",
        marker="*",
        s=100,
        label="Best Trial",
    )

    plt.xlabel("Reconstruction Error (RMAE)")
    plt.ylabel("KL Divergence")

    plt.legend()

    plt.savefig(config.PATHS["figures_dir"] / "optimization_results_utopia.png")
    plt.show()

    # %%

    print(df.index[best_index])
