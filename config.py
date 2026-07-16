from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"  # gitignored; user downloads Zenodo artifact here
FIGURES_DIR = ROOT / "figures_out"  # gitignored; where scripts write output figures

PATHS = {
    "segmented_data": DATA_DIR / "segmented_data.h5",
    "raw_data": DATA_DIR / "raw_data.h5",
    "pca": DATA_DIR / "pca_segmented.pkl",
    "vae": DATA_DIR / "vae.pkl",
    "scaler_scores": DATA_DIR / "scaler_scores.pkl",
    "scaler_params": DATA_DIR / "scaler_params.pkl",
    "figures_dir": FIGURES_DIR,
}

N_SCORES = 50

# Exact hyperparameters of run_246 (the paper's final model), transcribed
# from experiments/segmented_scores_vae_optimization/run_246/summary.yaml.
# That run recorded `seed: null`, so we pin seed=0 here for reproducibility
# of this public port (the original run did not fix a seed).
HYPERPARAMETERS = {
    "encoder_shape": (64,),
    "decoder_shape": (64,),
    "latent_dim": 32,
    "activation_function": "gelu",
    "optimizer": "adam",
    "lr": 0.008083354808047812,
    "beta": 0.0003359579661127939,
    "kl_warmup_epochs": 10,
    "kl_annealing_epochs": 100,
    "free_bits_threshold": 0.0,
    "scheduler": "cosine_annealing",
    "T_max": 300,
    "n_epochs": 300,
    "batch_size": 64,
    "batch_norm_momentum": None,
    "progressive_dropout": False,
    "dropout_ratio": 0.0,
    "early_stopping": False,
    "patience": 10,
    "min_delta": 0.0,
    "seed": 0,
}
