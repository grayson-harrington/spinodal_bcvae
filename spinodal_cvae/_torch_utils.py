# ---------------------------------------------------------------------------
# Standalone utility functions and classes.
#
# These are self-contained replacements for what the original (private lab
# package) implementation obtained from its shared model-utilities module and
# from its `BaseGenerativeModel` / `BaseTabularModel` / `BaseModel` inheritance
# chain. They are reproduced here so that `VariationalAutoencoder` and
# `InvertibleNeuralNetwork` are byte-for-byte behaviorally identical to the
# originals, with zero private-package dependency.
# ---------------------------------------------------------------------------

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_default_batch_size(n_samples: int) -> int:
    if n_samples < 32:
        return max(1, n_samples // 2)
    elif n_samples < 1000:
        return 16 if n_samples < 200 else 32
    elif n_samples < 10_000:
        return 64
    elif n_samples < 50_000:
        return 128
    else:
        return 256


def create_data_loader(
    *,
    X,
    y,
    batch_size,
    c=None,
    X_dtype=torch.float32,
    y_dtype=torch.float32,
    c_dtype=torch.float32,
    drop_last=True,
    seed=None,
):
    tensor_X = torch.tensor(X, dtype=X_dtype)
    tensor_y = torch.tensor(y, dtype=y_dtype)
    if c is not None:
        tensor_c = torch.tensor(c, dtype=c_dtype)
        dataset = TensorDataset(tensor_X, tensor_y, tensor_c)
    else:
        dataset = TensorDataset(tensor_X, tensor_y)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = torch.inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# --- activation / optimizer / scheduler / loss lookup tables --------------
# Only the subset actually used by this repo's final model + HPO search is
# implemented (gelu/relu/tanh/leaky_relu activations, adam/sgd optimizers,
# cosine_annealing/step/multistep schedulers, mse loss), following the exact
# lookup-and-override pattern of the original private-package utilities.

_activations = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "leaky_relu": torch.nn.LeakyReLU,
}
_activation_hyperparameters_and_defaults = {
    "gelu": {"approximate": "none"},
    "relu": {},
    "tanh": {},
    "leaky_relu": {"negative_slope": 0.01},
}

_optimizers = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}
_optimizer_hyperparameters_and_defaults = {
    "adam": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0},
    "sgd": {
        "lr": 1e-3,
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
    },
}

_schedulers = {
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
}
_scheduler_hyperparameters_and_defaults = {
    "cosine_annealing": {"T_max": 100, "eta_min": 1e-8, "last_epoch": -1},
    "step": {"step_size": None, "gamma": 0.1, "last_epoch": -1},
    "multistep": {"milestones": None, "gamma": 0.1, "last_epoch": -1},
}

_losses = {"mse": torch.nn.MSELoss}
_loss_hyperparameters_and_defaults = {"mse": {"reduction": "mean"}}


def get_activation_validate_kwargs(name, **kwargs):
    name = name.lower()
    final_kwargs = dict(_activation_hyperparameters_and_defaults[name])
    for k in kwargs:
        if k in final_kwargs:
            final_kwargs[k] = kwargs[k]
    return _activations[name], final_kwargs


def get_optimizer_validate_kwargs(name, **kwargs):
    name = name.lower()
    final_kwargs = dict(_optimizer_hyperparameters_and_defaults[name])
    for k in kwargs:
        if k in final_kwargs:
            final_kwargs[k] = kwargs[k]
    return _optimizers[name], final_kwargs


def get_scheduler_validate_kwargs(name, **kwargs):
    if not name:
        return None, {}
    name = name.lower()
    final_kwargs = dict(_scheduler_hyperparameters_and_defaults[name])
    for k in kwargs:
        if k in final_kwargs:
            final_kwargs[k] = kwargs[k]
    return _schedulers[name], final_kwargs


def get_loss_validate_kwargs(name, **kwargs):
    name = name.lower()
    final_kwargs = dict(_loss_hyperparameters_and_defaults[name])
    for k in kwargs:
        if k in final_kwargs:
            final_kwargs[k] = kwargs[k]
    return _losses[name], final_kwargs
