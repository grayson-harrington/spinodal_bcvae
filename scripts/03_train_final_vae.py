from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import h5py
import json
import yaml
import pickle

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spinodal_cvae import VariationalAutoencoder
from spinodal_cvae.metrics import mae

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from src_plotting._final_vae_utils import (
    plot_losses,
    plot_parity_plots,
    get_relative_MAE_per_score,
    plot_relative_MAE_vs_score,
    get_KL_vs_latent_dim,
    plot_KL_vs_latent_dim,
    plot_distribution_fidelity_pc_histograms,
    pick_example_indices,
    plot_selected_recon_examples,
    plot_vae_sample_cross_sections,
    sweep_one_condition,
    plot_pc_trend_with_variance_multi,
)

import config


def trim_pca(pca, n_components):
    pca.components_ = pca.components_[:n_components]
    pca.explained_variance_ = pca.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca.singular_values_[:n_components]
    pca.n_components_ = n_components
    return pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams-yaml",
        type=str,
        default=None,
        help="Optional path to a run summary YAML with ['params']['vae'] hyperparameters. "
        "Defaults to config.HYPERPARAMETERS if not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the VAE random seed. If given, overrides the "
        "'seed' entry in the hyperparameters. Used by 04_epistemic_ensemble.py to "
        "produce genuinely independently-seeded ensemble members.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional override for the trained-model output path. Defaults to "
        "config.PATHS['vae']. Scaler pickles are always written to their config paths.",
    )
    args = parser.parse_args()

    path_data = config.PATHS["segmented_data"]
    path_data_raw = config.PATHS["raw_data"]
    n_scores = config.N_SCORES

    if args.hparams_yaml is not None:
        with open(args.hparams_yaml, "rb") as f:
            best_hyperparameters = yaml.safe_load(f)["params"]["vae"]
    else:
        best_hyperparameters = config.HYPERPARAMETERS

    # Copy so a --seed override never mutates the shared config dict.
    best_hyperparameters = dict(best_hyperparameters)
    if args.seed is not None:
        best_hyperparameters["seed"] = args.seed

    out_vae_path = args.out if args.out is not None else config.PATHS["vae"]

    print(json.dumps(best_hyperparameters, indent=4))

    with h5py.File(path_data_raw, "r") as f:
        parameter_names = [name.decode("utf-8") for name in f["parameter_names"][...]]

    with open(config.PATHS["pca"], "rb") as f:
        pca: PCA = pickle.load(f)
        pca = trim_pca(pca, n_scores)

    with h5py.File(path_data, "r") as f:
        scores_train = f["hetero_only"]["train"]["scores"][:, :n_scores]
        params_train = f["hetero_only"]["train"]["params"][...]
        micros_train = f["hetero_only"]["train"]["micros"][...]
        stats_train = f["hetero_only"]["train"]["stats"][..., 0]
        stats_train_trunc = pca.inverse_transform(scores_train)
        stats_train_trunc = stats_train_trunc.reshape(
            -1, *tuple([int(stats_train_trunc.shape[-1] ** 0.5)] * 2)
        )

        scores_validate = f["hetero_only"]["validate"]["scores"][:, :n_scores]
        params_validate = f["hetero_only"]["validate"]["params"][...]
        micros_validate = f["hetero_only"]["validate"]["micros"][...]
        stats_validate = f["hetero_only"]["validate"]["stats"][..., 0]
        stats_validate_trunc = pca.inverse_transform(scores_validate)
        stats_validate_trunc = stats_validate_trunc.reshape(
            -1, *tuple([int(stats_validate_trunc.shape[-1] ** 0.5)] * 2)
        )

        scores_test = f["hetero_only"]["test"]["scores"][:, :n_scores]
        params_test = f["hetero_only"]["test"]["params"][...]
        micros_test = f["hetero_only"]["test"]["micros"][...]
        stats_test = f["hetero_only"]["test"]["stats"][..., 0]
        stats_test_trunc = pca.inverse_transform(scores_test)
        stats_test_trunc = stats_test_trunc.reshape(
            -1, *tuple([int(stats_test_trunc.shape[-1] ** 0.5)] * 2)
        )

        print("Train")
        print(scores_train.shape, params_train.shape)
        print(micros_train.shape, stats_train.shape)

        print("Validate")
        print(scores_validate.shape, params_validate.shape)
        print(micros_validate.shape, stats_validate.shape)

        print("Test")
        print(scores_test.shape, params_test.shape)
        print(micros_test.shape, stats_test.shape)

    ######## DATA PREPROCESSING ########

    scores_train = np.concatenate((scores_train, scores_validate), axis=0)
    params_train = np.concatenate((params_train, params_validate), axis=0)
    micros_train = np.concatenate((micros_train, micros_validate), axis=0)
    stats_train = np.concatenate((stats_train, stats_validate), axis=0)
    stats_train_trunc = np.concatenate(
        (stats_train_trunc, stats_validate_trunc), axis=0
    )

    scaler_scores = MinMaxScaler(feature_range=(-1, 1), clip=True).fit(scores_train)
    scaler_params = StandardScaler().fit(params_train)

    pickle.dump(scaler_scores, open(config.PATHS["scaler_scores"], "wb"))
    pickle.dump(scaler_params, open(config.PATHS["scaler_params"], "wb"))

    scores_train_scaled = scaler_scores.transform(scores_train)
    scores_test_scaled = scaler_scores.transform(scores_test)

    params_train_scaled = scaler_params.transform(params_train)
    params_test_scaled = scaler_params.transform(params_test)

    ######## MODEL TRAINING ########

    print(json.dumps(best_hyperparameters, indent=4))

    model = VariationalAutoencoder(**best_hyperparameters)

    print(json.dumps(model.hyperparameters, indent=4, default=str))

    model.fit(
        X_train=scores_train_scaled,
        y_train=scores_train_scaled,
        c_train=params_train_scaled,
        X_validation=scores_test_scaled,
        y_validation=scores_test_scaled,
        c_validation=params_test_scaled,
    )

    pickle.dump(model, open(out_vae_path, "wb"))

    plot_losses(model.losses, show_plot=False)

    # make reconstruction predictions for train and test

    scores_train_recon_scaled = model.predict(scores_train_scaled, params_train_scaled)
    scores_test_recon_scaled = model.predict(scores_test_scaled, params_test_scaled)

    scores_train_recon = scaler_scores.inverse_transform(scores_train_recon_scaled)
    scores_test_recon = scaler_scores.inverse_transform(scores_test_recon_scaled)

    plot_parity_plots(
        scores_train,
        scores_train_recon,
        scores_test,
        scores_test_recon,
        n_pcs=8,
        show_plot=False,
    )

    plot_parity_plots(
        scores_train,
        scores_train_recon,
        scores_test,
        scores_test_recon,
        n_pcs=50,
        show_plot=False,
    )

    rel_mae_train, rel_mae_test = get_relative_MAE_per_score(
        scores_train, scores_train_recon, scores_test, scores_test_recon
    )

    print(f"RMAE_train: {rel_mae_train}")
    print(f"val RMAE (average_RMAE_train): {float(rel_mae_train.mean())}")
    print(f"RMAE_test: {rel_mae_test}")
    print(f"val RMAE (average_RMAE_test): {float(rel_mae_test.mean())}")

    rel_mae_train, rel_mae_validate = plot_relative_MAE_vs_score(
        rel_mae_train,
        rel_mae_test,
        show_plot=False,
    )

    # get stat recontruction predictions for test
    scores_test_recon_pred = model.predict(
        X=scaler_scores.transform(scores_test),
        c=scaler_params.transform(params_test),
    )
    scores_test_recon_pred = scaler_scores.inverse_transform(scores_test_recon_pred)

    # latent histograms should be unit normal (encode then reparameterize)
    z_mu_train, z_logvar_train = model.encode(
        X=scores_train_scaled, c=params_train_scaled
    )
    z_mu_test, z_logvar_test = model.encode(X=scores_test_scaled, c=params_test_scaled)

    z_std_train = np.exp(0.5 * z_logvar_train)
    z_std_test = np.exp(0.5 * z_logvar_test)

    z_train = z_mu_train + z_std_train * np.random.randn(*z_mu_train.shape)
    z_test = z_mu_test + z_std_test * np.random.randn(*z_mu_test.shape)

    kl_train, kl_test = get_KL_vs_latent_dim(
        z_mu_train, z_logvar_train, z_mu_test, z_logvar_test
    )
    plot_KL_vs_latent_dim(kl_train, kl_test, show_plot=False)

    avg_kl_train = float(kl_train.mean())
    avg_kl_test = float(kl_test.mean())
    print(f"average_KL_train: {avg_kl_train}")
    print(f"val KL (average_KL_test): {avg_kl_test}")

    n_samples_per_val = 1000
    n_val = len(params_validate)
    n_total = n_val * n_samples_per_val

    # Sample from latent prior
    latent_samples = np.random.randn(n_total, model.latent_dim)

    # Repeat validation params for conditioning
    params_validate_expanded = np.repeat(params_test_scaled, n_samples_per_val, axis=0)

    # Decode
    sampled_scores_scaled = model.decode(z=latent_samples, c=params_validate_expanded)
    sampled_scores = scaler_scores.inverse_transform(
        sampled_scores_scaled
    )  # shape: (n_total, n_scores)

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    n_samples_per_val = 1000
    n_val = len(params_test_scaled)
    latent_dim = model.latent_dim

    # Expand conditioning parameters
    params_expanded = np.repeat(params_test_scaled, n_samples_per_val, axis=0)

    # ------------------------------------------------------------
    # 1) Posterior predictive (reconstruction) sampling
    #    encode -> sample -> decode
    # ------------------------------------------------------------
    # Encode test data
    z_mu, z_logvar = model.encode(
        X=scores_test_scaled,
        c=params_test_scaled,
    )

    # Reparameterization trick
    z_std = np.exp(0.5 * z_logvar)

    # Sample multiple latent draws per test point
    eps = np.random.randn(n_val, n_samples_per_val, latent_dim)
    z_post = (z_mu[:, None, :] + z_std[:, None, :] * eps).reshape(-1, latent_dim)

    # Decode posterior samples
    recon_scores_scaled = model.decode(z=z_post, c=params_expanded)
    recon_scores = scaler_scores.inverse_transform(recon_scores_scaled)

    # ------------------------------------------------------------
    # 2) Conditional generative sampling
    #    sample from prior -> decode
    # ------------------------------------------------------------
    z_prior = np.random.randn(n_val * n_samples_per_val, latent_dim)

    gen_scores_scaled = model.decode(z=z_prior, c=params_expanded)
    gen_scores = scaler_scores.inverse_transform(gen_scores_scaled)

    plot_distribution_fidelity_pc_histograms(
        reference_pc=scores_test,  # empirical reference
        reconstructed_pc=recon_scores,  # posterior predictive
        generated_pc=gen_scores,  # prior-sampled generation
        show_plot=False,
        label_reference="Test Set",
        label_reconstructed="Test Set Reconstruction",
        label_generated=r"$\beta$-cVAE Sample Generation",
    )

    # compute mae_test_stats based on the average error from many samples. Use this to pick examples.

    # Get a lot of VAE samples for each test sample's params
    n_vae = 1000
    vae_samples_scaled = []

    for c in scaler_params.transform(params_test):
        # Repeat each c n_vae times for sampling
        c_repeat = np.repeat(c[np.newaxis, :], n_vae, axis=0)
        vae_sampled_scores = model.sample(num_samples=n_vae, c=c_repeat)
        vae_samples_scaled.append(vae_sampled_scores)

    vae_samples_scaled = np.stack(vae_samples_scaled)

    # Reconstruct to PC space using score_scaler
    vae_samples_unscaled = scaler_scores.inverse_transform(
        vae_samples_scaled.reshape(-1, vae_samples_scaled.shape[-1])
    )
    vae_samples_unscaled = vae_samples_unscaled.reshape(len(scores_test), n_vae, -1)

    # Inverse transform from PCs to stats/images for both val and VAE samples

    s = vae_samples_unscaled.shape

    temp = vae_samples_unscaled.reshape(-1, vae_samples_unscaled.shape[-1])
    n_batches = s[0]

    test_stats_recon_pred = []
    test_stats_mmae = []
    for i, (vae_batch, test_stat) in enumerate(
        zip(np.array_split(temp, n_batches), stats_test)
    ):
        # process the stats, get mae per sample, get mean of maes (expected mae), delete bigs to conserve memory.

        vae_stats = pca.inverse_transform(vae_batch)
        vae_stats = np.reshape(
            vae_stats, (s[1], test_stat.shape[-2], test_stat.shape[-1])
        )
        sample_maes = mae(test_stat[None, ...], vae_stats, axis=(1, 2))

        mean_mae = np.mean(sample_maes)
        test_stats_mmae.append(mean_mae)

        # get recon sample that is closest to mean
        closest = np.argmin(np.abs(sample_maes - mean_mae))
        test_stats_recon_pred.append(vae_stats[closest].copy())

        del vae_stats
        del sample_maes

        if (i + 1) % 5 == 0:
            print(f"{i + 1}/{n_batches}")

    test_stats_mmae = np.array(test_stats_mmae)
    test_stats_recon_pred = np.stack(test_stats_recon_pred)

    print()
    print(test_stats_mmae.shape)
    print(test_stats_recon_pred.shape)

    # log sampling error metrics

    print(f"mmae_test_stats_sampling_error: {test_stats_mmae}")

    # Persist the per-test-sample mean-MAE sampling error so the optional
    # epistemic-uncertainty analysis (04b_epistemic_uncertainty.py) can reuse
    # the exact same example selection.
    np.save(
        config.DATA_DIR / "metrics__analytics__mmae_test_stats_sampling_error.npy",
        test_stats_mmae,
    )

    selected_inds = pick_example_indices(test_stats_mmae)

    # Example 1: Min NMAE
    # Examples 2-4: Near Mean NMAE
    # Example 5: Max NMAE

    for i, ind in enumerate(selected_inds):
        print(f"mmae_test_stats_sampling_error_E{i + 1}_{ind}: {test_stats_mmae[ind]}")
        print(f"selected_test_sample_conditions_E{i + 1}_{ind}: {params_test[ind]}")

    plot_selected_recon_examples(
        stats_test,
        stats_test_trunc,
        test_stats_recon_pred,
        indices=selected_inds,
        show_plot=False,
    )

    # resample VAE 1000 times for the selected indices...

    stats_test_selected = stats_test[selected_inds]
    scores_selected = pca.transform(
        stats_test_selected.reshape(stats_test_selected.shape[0], -1)
    )
    c_selected = params_test[selected_inds]

    # Get a lot of VAE samples for each test sample's params
    n_vae = 1000
    vae_samples_scaled = []

    for c in scaler_params.transform(c_selected):
        # Repeat each c n_vae times for sampling
        c_repeat = np.repeat(c[np.newaxis, :], n_vae, axis=0)
        vae_sampled_scores = model.sample(num_samples=n_vae, c=c_repeat)
        vae_samples_scaled.append(vae_sampled_scores)

    vae_samples_scaled = np.stack(vae_samples_scaled)

    # Reconstruct to PC space using score_scaler
    vae_samples_unscaled = scaler_scores.inverse_transform(
        vae_samples_scaled.reshape(-1, vae_samples_scaled.shape[-1])
    )
    vae_samples_unscaled = vae_samples_unscaled.reshape(
        len(stats_test_selected), n_vae, -1
    )

    # Compare a few sample reconstructions to the original validation reconstruction (stats/image)
    # Inverse transform from PCs to stats/images for both val and VAE samples

    s = vae_samples_unscaled.shape

    temp = vae_samples_unscaled.reshape(-1, vae_samples_unscaled.shape[-1])

    test_vae_stats_selected = pca.inverse_transform(temp)
    test_vae_stats_selected = np.reshape(
        test_vae_stats_selected, (s[0], s[1], test_stat.shape[-2], test_stat.shape[-1])
    )

    plot_vae_sample_cross_sections(
        stats_test_selected, test_vae_stats_selected, show_plot=False
    )

    # ---------- Config ----------
    n_per_cond = 100000  # samples per sweep value
    n_sweep = 25  # number of sweep points per condition
    batch_size = 20000  # sampling batch size to control memory; tune as needed
    cond_indices = list(range(18))  # which condition dims to sweep (0..17)
    # Optional: names for pretty titles/labels

    with h5py.File(config.PATHS["raw_data"], "r") as f:
        print(f.keys())
        print([s.decode("utf-8") for s in f["parameter_names"][...]])

        condition_names = [s.decode("utf-8") for s in f["parameter_names"][...]]

    # ---- Select representative samples (unchanged) ----

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(params_test)

    closest_indices, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, params_test
    )
    selected_points = params_test[closest_indices]

    # ---- One plot per condition ----
    for idx in cond_indices:
        all_mean_pc = []
        all_std_pc = []

        for params in selected_points:
            sweep_vals, mean_pc, std_pc = sweep_one_condition(
                cond_idx=idx,
                base_param=params,
                params_train=params_train,
                params_scaler=scaler_params,
                score_scaler=scaler_scores,
                model=model,
                n_sweep=n_sweep,
                n_per_cond=n_per_cond,
                batch_size=batch_size,
            )

            all_mean_pc.append(mean_pc)
            all_std_pc.append(std_pc)

        all_mean_pc = np.stack(all_mean_pc, axis=0)
        all_std_pc = np.stack(all_std_pc, axis=0)

        cname = (
            condition_names[idx]
            if (condition_names and idx < len(condition_names))
            else f"Condition {idx}"
        )

        plot_pc_trend_with_variance_multi(
            x_values=sweep_vals,
            mean_pcs=all_mean_pc,
            std_pcs=all_std_pc,
            title=f"PC Trends with Variance — {cname} Sweep",
            x_label=cname,
            show_plot=False,
            n_pcs=5,
        )

    for i, params in enumerate(selected_points):
        print(f"selected_test_sample_conditions_{i}_sweep: {selected_points}")
