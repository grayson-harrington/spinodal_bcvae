"""
Data-scaling study: beta-cVAE vs. conditional INN vs. Gaussian baseline.

Trains each model at N in {50, 100, 500, 1000} samples for 5 seeds, evaluates
with the sliced Wasserstein distance against a fixed held-out test set, and
saves a scaling figure plus a first-5-PC corner comparison. This reproduces
Appendix B of the manuscript.

Run from the repository root:

    python scripts/05_scaling_study.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time
import warnings

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from spinodal_cvae import InvertibleNeuralNetwork, VariationalAutoencoder

import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- config -----

N_SCORES = config.N_SCORES
TRAINING_SIZES = [50, 100, 500, 1000]
SEEDS = [0, 1, 2, 3, 4]
K_SAMPLES = 300
N_SWD_PROJECTIONS = 500
OUTPUT_DIR = config.FIGURES_DIR / "scaling_study"

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

# --------------------------------------------------------------- data --------

with h5py.File(config.PATHS["segmented_data"], "r") as f:
    scores_train = f["hetero_only"]["train"]["scores"][:, :N_SCORES]
    params_train = f["hetero_only"]["train"]["params"][...]
    scores_validate = f["hetero_only"]["validate"]["scores"][:, :N_SCORES]
    params_validate = f["hetero_only"]["validate"]["params"][...]
    scores_test = f["hetero_only"]["test"]["scores"][:, :N_SCORES]
    params_test = f["hetero_only"]["test"]["params"][...]

scores_pool = np.concatenate([scores_train, scores_validate], axis=0)
params_pool = np.concatenate([params_train, params_validate], axis=0)

print(f"Pool:  scores={scores_pool.shape}  params={params_pool.shape}")
print(f"Test:  scores={scores_test.shape}   params={params_test.shape}")

# beta-cVAE hyperparameters: the paper's final model, from config.py.
vae_params = dict(config.HYPERPARAMETERS)
print(json.dumps({k: v for k, v in vae_params.items()}, indent=4, default=str))

# ------------------------------------------------------------ utilities ------


def fit_scalers(scores_sub, params_sub):
    scaler_scores = MinMaxScaler(feature_range=(-1, 1), clip=True).fit(scores_sub)
    scaler_params = StandardScaler().fit(params_sub)
    return scaler_scores, scaler_params


def sliced_wasserstein_distance(X, Y, n_projections=200, seed=None):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    X_proj = X @ directions.T
    Y_proj = Y @ directions.T
    distances = [
        wasserstein_distance(X_proj[:, i], Y_proj[:, i]) for i in range(n_projections)
    ]
    return float(np.mean(distances))


def evaluate_model(
    model, scaler_scores, params_test_scaled, scores_test, k_samples, seed, n_projections=200
):
    params_expanded = np.repeat(params_test_scaled, k_samples, axis=0)
    gen_scaled = model.sample(c=params_expanded)
    gen_scores = scaler_scores.inverse_transform(gen_scaled)
    return sliced_wasserstein_distance(
        scores_test, gen_scores, n_projections=n_projections, seed=seed
    )


# ------------------------------------------------- model builders -----------


def build_vae(seed, vae_params):
    params = {**vae_params, "seed": seed}
    return VariationalAutoencoder(**params)


def build_inn(seed):
    return InvertibleNeuralNetwork(
        n_blocks=2,
        block_hidden_shape=26,
        activation_function="gelu",
        clamp=2.0,
        gin_block=False,
        permute_soft=False,
        optimizer="adam",
        scheduler="cosine_annealing",
        T_max=300,
        n_epochs=300,
        batch_size=64,
        early_stopping=True,
        patience=10,
        min_delta=0.0,
        seed=seed,
    )


class MarginalGaussianBaseline:
    def __init__(self, seed=None):
        self.seed = seed
        self.mu = None
        self.sigma = None

    def fit(self, scores_scaled):
        self.mu = scores_scaled.mean(axis=0)
        self.sigma = scores_scaled.std(axis=0) + 1e-8
        return self

    def sample(self, c):
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((len(c), len(self.mu))) * self.sigma + self.mu


# --------------------------------------------------------- scaling study ----

results = {
    model_type: {N: {"swd": [], "time": []} for N in TRAINING_SIZES}
    for model_type in ("vae", "inn", "gaussian")
}

for N in TRAINING_SIZES:
    for seed in SEEDS:
        print(f"\nN={N:5d} | seed={seed}")

        rng = np.random.default_rng(seed * 10000 + N)
        idx = rng.choice(len(scores_pool), size=N, replace=False)
        scores_sub = scores_pool[idx]
        params_sub = params_pool[idx]

        scaler_scores, scaler_params = fit_scalers(scores_sub, params_sub)

        scores_sub_scaled = scaler_scores.transform(scores_sub)
        params_sub_scaled = scaler_params.transform(params_sub)
        scores_test_scaled = scaler_scores.transform(scores_test)
        params_test_scaled = scaler_params.transform(params_test)

        # --- VAE ---
        print("  Training VAE...")
        vae = build_vae(seed, vae_params)
        t0 = time.time()
        vae.fit(
            X_train=scores_sub_scaled,
            y_train=scores_sub_scaled,
            c_train=params_sub_scaled,
            X_validation=scores_test_scaled,
            y_validation=scores_test_scaled,
            c_validation=params_test_scaled,
        )
        vae_time = time.time() - t0
        n_vae = sum(p.numel() for p in vae.encoder.parameters()) + sum(
            p.numel() for p in vae.decoder.parameters()
        )
        print(f"  VAE parameters: {n_vae:,}")
        vae_swd = evaluate_model(
            vae, scaler_scores, params_test_scaled, scores_test, K_SAMPLES, seed, N_SWD_PROJECTIONS
        )
        results["vae"][N]["swd"].append(vae_swd)
        results["vae"][N]["time"].append(vae_time)

        # --- INN ---
        print("  Training INN...")
        inn = build_inn(seed)
        t0 = time.time()
        try:
            inn.fit(
                X_train=scores_sub_scaled,
                y_train=scores_sub_scaled,
                c_train=params_sub_scaled,
                X_validation=scores_test_scaled,
                y_validation=scores_test_scaled,
                c_validation=params_test_scaled,
            )
            print(f"  INN parameters: {inn.n_net_parameters:,}")
            inn_time = time.time() - t0
            inn_swd = evaluate_model(
                inn, scaler_scores, params_test_scaled, scores_test, K_SAMPLES, seed, N_SWD_PROJECTIONS
            )
        except Exception as e:
            print(f"  INN failed (seed={seed}, N={N}): {e}")
            inn_time = float("nan")
            inn_swd = float("nan")
        results["inn"][N]["swd"].append(inn_swd)
        results["inn"][N]["time"].append(inn_time)

        # --- Gaussian baseline ---
        gaussian = MarginalGaussianBaseline(seed=seed).fit(scores_sub_scaled)
        gaussian_swd = evaluate_model(
            gaussian, scaler_scores, params_test_scaled, scores_test, K_SAMPLES, seed, N_SWD_PROJECTIONS
        )
        results["gaussian"][N]["swd"].append(gaussian_swd)
        results["gaussian"][N]["time"].append(0.0)

        print(f"  SWD -> vae={vae_swd:.4f}  inn={inn_swd:.4f}  gaussian={gaussian_swd:.4f}")

# --------------------------------------------------------- save results -----

os.makedirs(OUTPUT_DIR, exist_ok=True)

json_safe = {}
for model_type, n_dict in results.items():
    json_safe[model_type] = {}
    for N, run_data in n_dict.items():
        json_safe[model_type][str(N)] = {
            k: [float(v) for v in vals] for k, vals in run_data.items()
        }

json_path = os.path.join(OUTPUT_DIR, "scaling_results.json")
with open(json_path, "w") as f:
    json.dump(json_safe, f, indent=2)

np.savez(os.path.join(OUTPUT_DIR, "scaling_results.npz"), results=results)
print(f"Results saved to {json_path}")

# ------------------------------------------------------- plot figure --------

fig, ax = plt.subplots(figsize=(4.5, 3.5))

styles = {
    "vae": {"color": "tab:blue", "marker": "o", "ls": "-", "label": r"$\beta$-cVAE"},
    "inn": {"color": "tab:orange", "marker": "s", "ls": "-", "label": "cINN"},
    "gaussian": {"color": "tab:gray", "marker": None, "ls": "--", "label": "Gaussian baseline"},
}

for model_type, style in styles.items():
    means, ses = [], []
    for N in TRAINING_SIZES:
        vals = [v for v in results[model_type][N]["swd"] if not np.isnan(v)]
        if len(vals) == 0:
            means.append(np.nan)
            ses.append(np.nan)
        else:
            means.append(np.mean(vals))
            ses.append(np.std(vals) / np.sqrt(len(vals)))

    means = np.array(means)
    ses = np.array(ses)

    ax.plot(
        TRAINING_SIZES,
        means,
        color=style["color"],
        linestyle=style["ls"],
        marker=style["marker"],
        label=style["label"],
    )
    ax.fill_between(
        TRAINING_SIZES,
        means - ses,
        means + ses,
        color=style["color"],
        alpha=0.2,
    )

ax.set_xscale("log")
ax.set_xticks(TRAINING_SIZES)
ax.set_xticklabels([str(N) for N in TRAINING_SIZES])
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Sliced Wasserstein Distance")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()

figure_path = os.path.join(OUTPUT_DIR, "figure_scaling_study.png")
fig.savefig(figure_path)
print(f"Figure saved to {figure_path}")

# ------------------------------------------------------- corner plots -------
# One plot per model type at N=1000, seed=0; each overlays the test set with
# generated samples.

import corner
import matplotlib.lines as mlines

N_CORNER = 1000
SEED_CORNER = 0
N_CORNER_SAMPLES = 5000
N_PC_CORNER = 5

rng_c = np.random.default_rng(SEED_CORNER * 10000 + N_CORNER)
idx_c = rng_c.choice(len(scores_pool), size=N_CORNER, replace=False)
scaler_scores_c, scaler_params_c = fit_scalers(scores_pool[idx_c], params_pool[idx_c])

scores_sub_c = scaler_scores_c.transform(scores_pool[idx_c])
params_sub_c = scaler_params_c.transform(params_pool[idx_c])
scores_test_c = scaler_scores_c.transform(scores_test)
params_test_c = scaler_params_c.transform(params_test)

vae_c = build_vae(SEED_CORNER, vae_params)
vae_c.fit(
    X_train=scores_sub_c, y_train=scores_sub_c, c_train=params_sub_c,
    X_validation=scores_test_c, y_validation=scores_test_c, c_validation=params_test_c,
)

inn_c = build_inn(SEED_CORNER)
inn_c.fit(
    X_train=scores_sub_c, y_train=scores_sub_c, c_train=params_sub_c,
    X_validation=scores_test_c, y_validation=scores_test_c, c_validation=params_test_c,
)

gaussian_c = MarginalGaussianBaseline(seed=SEED_CORNER).fit(scores_sub_c)

params_expanded_c = np.repeat(params_test_c, N_CORNER_SAMPLES // len(params_test_c), axis=0)

gen_vae = scaler_scores_c.inverse_transform(vae_c.sample(c=params_expanded_c))
gen_inn = scaler_scores_c.inverse_transform(inn_c.sample(c=params_expanded_c))
gen_gaussian = scaler_scores_c.inverse_transform(gaussian_c.sample(c=params_expanded_c))

COLOR_REF = "tab:blue"
COLOR_GEN = "tab:orange"

corner_kwargs = dict(
    bins=30,
    smooth=2.0,
    smooth1d=1.0,
    labels=[f"PC{i+1}" for i in range(N_PC_CORNER)],
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    levels=(0.393, 0.864),
    label_kwargs={"fontsize": 14},
)

panel_size = 2.5
fig_size = panel_size * N_PC_CORNER

corner_paths = []
for model_name, gen_scores_c in [
    (r"$\beta$-cVAE", gen_vae),
    ("cINN", gen_inn),
    ("Gaussian baseline", gen_gaussian),
]:
    rng_sub = np.random.default_rng(42)
    max_pts = min(5000, len(gen_scores_c))

    def _sub(arr):
        return arr[rng_sub.choice(len(arr), max_pts, replace=len(arr) < max_pts)]

    fig = corner.corner(
        _sub(scores_test)[:, :N_PC_CORNER],
        color=COLOR_REF,
        fig=plt.figure(figsize=(fig_size, fig_size)),
        **corner_kwargs,
    )
    corner.corner(
        _sub(gen_scores_c)[:, :N_PC_CORNER],
        color=COLOR_GEN,
        fig=fig,
        hist_kwargs={"linestyle": "--"},
        contour_kwargs={"linestyles": ["--", "--"]},
        **corner_kwargs,
    )

    for ax in fig.axes:
        ax.tick_params(labelsize=12)

    axs = np.array(fig.axes).reshape(N_PC_CORNER, N_PC_CORNER)
    legend_ax = axs[1, N_PC_CORNER - 2]
    legend_ax.set_visible(True)
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[
            mlines.Line2D([], [], color=COLOR_REF, label="Test set"),
            mlines.Line2D([], [], color=COLOR_GEN, label=model_name, linestyle="--"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=14,
    )

    fig.suptitle(model_name, fontsize=16, y=1.01)
    safe_name = model_name.replace("$", "").replace("\\", "").replace(" ", "_").replace("-", "")
    out = os.path.join(OUTPUT_DIR, f"corner_{safe_name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    corner_paths.append(out)
    print(f"Saved -> {out}")

# The three corner PNGs above are Appendix B Figs. B.2 (beta-cVAE), B.3 (cINN),
# and B.4 (Gaussian baseline).
