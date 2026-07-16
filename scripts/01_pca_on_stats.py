from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import config


if __name__ == "__main__":
    path_data = config.PATHS["segmented_data"]

    with h5py.File(path_data, "r") as f:
        print(f.keys())

        stats_train = f["hetero_only"]["train"]["stats"][...]
        stats_validate = f["hetero_only"]["validate"]["stats"][...]
        stats_test = f["hetero_only"]["test"]["stats"][...]

        print(stats_train.shape, stats_validate.shape, stats_test.shape)

    pca = PCA()
    pca.fit(stats_train.reshape(stats_train.shape[0], -1))

    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    config.PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)
    plt.savefig(config.PATHS["figures_dir"] / "pca_cumulative_explained_variance.png")

    with open(config.PATHS["pca"], "wb") as f:
        pickle.dump(pca, f)

    print(f"Saved fitted PCA -> {config.PATHS['pca']}")

    # NOTE: The PC scores for the train/validate/test splits are already
    # pre-computed and stored inside the deposited segmented_data.h5 artifact
    # (under hetero_only/<split>/scores). Downstream scripts (03_train_final_vae.py,
    # the figure scripts) read those scores directly, so we deliberately do NOT
    # write scores back into the h5 file here — segmented_data.h5 is treated as
    # a read-only artifact. This script's only outputs are the fitted PCA pickle
    # above and the cumulative-explained-variance figure.
    #
    # For reference, the scores would be produced as:
    #   scores_<split> = pca.transform(stats_<split>.reshape(n, -1))
    scores_train = pca.transform(stats_train.reshape(stats_train.shape[0], -1))
    print(f"scores_train shape (recomputed, not written back): {scores_train.shape}")
