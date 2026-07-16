# spinodal-cvae

A self-contained, public implementation of the PCA → conditional β-VAE pipeline
used to build a fast generative surrogate for spinodal-decomposition
microstructures (represented by their two-point statistics). This repository
accompanies a manuscript submitted to *Computational Materials Science*
(Elsevier). It depends only on public Python packages — no private lab code is
required to reproduce the results.

The pipeline reduces two-point-statistics fields to principal-component (PC)
scores via PCA, then trains a conditional variational autoencoder (cVAE) on
those scores conditioned on the 18 physical simulation parameters, enabling
conditional generation, reconstruction, and latent-space analysis of
microstructural statistics.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 is recommended (the pinned versions in `requirements.txt` were
validated on 3.10).

## Data & model download

The processed dataset and trained artifacts are hosted on Zenodo (DOI: **TBD**).
Download the artifact and place its contents into the `data/` directory
(gitignored) so the layout is:

```
data/
├── segmented_data.h5     # train/validate/test scores, params, micros, stats under hetero_only/
├── raw_data.h5           # parameter_names
├── pca_segmented.pkl     # fitted PCA (regenerate with 01_ if desired)
├── vae.pkl               # trained conditional VAE
├── scaler_scores.pkl     # MinMaxScaler for PC scores
└── scaler_params.pkl     # StandardScaler for conditioning parameters
```

`segmented_data.h5` is treated as a **read-only** artifact: its
`hetero_only/<split>/scores` datasets are already the PCA scores, so no script
writes back into it.

If you only want to *use* the trained model, you need `vae.pkl`, the two scaler
pickles, `pca_segmented.pkl`, and `segmented_data.h5`. To retrain from scratch
you also need to run the pipeline below.

## Run order

Run scripts from the repository root (e.g. `python scripts/01_pca_on_stats.py`).

Main pipeline:

1. `scripts/01_pca_on_stats.py` — fit PCA on the training two-point statistics
   and save `data/pca_segmented.pkl` plus a cumulative-explained-variance figure.
2. `scripts/03_train_final_vae.py` — train the final conditional VAE with the
   paper's hyperparameters (from `config.HYPERPARAMETERS`), save
   `data/vae.pkl` + the two scaler pickles, and produce the training/analysis
   figures. Prints validation RMAE and KL.
3. `scripts/figures/*.py` — regenerate the manuscript's latent-space and
   distribution-fidelity figures from the trained artifacts.

Optional / reproduce-from-scratch (not required for the main pipeline):

- `scripts/02_optimize_vae.py` — the multi-objective (MOTPE) Optuna
  hyperparameter search that selected the final hyperparameters.
- `scripts/04_epistemic_ensemble.py` — retrains the 11-model seeded ensemble
  and runs the epistemic-vs-aleatoric uncertainty analysis (Fig. 11). The
  ensemble checkpoints are not shipped; this script regenerates them from seeds.

Output figures are written to `figures_out/` (gitignored).

## Package

`spinodal_cvae/` is the importable package that replaces the private lab
dependency:

- `spinodal_cvae.VariationalAutoencoder` — the MLP conditional β-VAE (KL
  warmup/annealing, free-bits, optional progressive dropout), self-contained
  and depending only on PyTorch.
- `spinodal_cvae.metrics.mae` — mean absolute error with an optional `axis`.

## Citation

Please cite the accompanying paper once published:

```bibtex
@article{harrington_spinodal_cvae,
  title   = {TODO: fill in title once published},
  author  = {Harrington, Grayson and others},
  journal = {Computational Materials Science},
  year    = {2026},
  note    = {TODO: volume/pages/DOI once published}
}
```

(Placeholder — please fill in once the paper is published.)

## License

MIT — see [LICENSE](LICENSE).
