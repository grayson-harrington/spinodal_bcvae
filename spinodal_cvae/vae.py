# external imports
import itertools
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Standalone utility functions and classes.
#
# These are self-contained replacements for what the original (private lab
# package) implementation obtained from its shared model-utilities module and
# from its `BaseGenerativeModel` / `BaseTabularModel` / `BaseModel` inheritance
# chain. They are reproduced here so that `VariationalAutoencoder` below is
# byte-for-byte behaviorally identical to the original, with zero private-
# package dependency.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# VariationalAutoencoder and its supporting nn.Module classes.
#
# Ported from the original private lab package's generative VAE module. Core
# VAE math (encoder/decoder architecture, reparameterization, loss =
# MSE(x_recon, x) + beta(epoch)*KL with free-bits clamp, beta warmup/anneal
# schedule, cosine-annealed Adam, per-epoch seeded shuffling, sample-weighted
# epoch-loss averaging) is kept verbatim. What changed: the class no longer
# inherits from BaseGenerativeModel -> BaseTabularModel -> BaseModel; instead a
# minimal self-contained base (hyperparameters dict, is_fit flag, thin fit/
# predict/sample wrappers) is inlined directly below, and calls to now-removed
# private-package utility functions are replaced with the local equivalents above.
# ---------------------------------------------------------------------------


class VariationalAutoencoder:
    """
    VariationalAutoencoder is a deep generative surrogate model based on the Variational Autoencoder (VAE) architecture.

    This class provides functionality to model complex, high-dimensional datasets through a latent variable representation.
    It supports fully customizable encoder and decoder architectures, KL-divergence annealing, optional dropout and batch
    normalization, and flexible optimizer/scheduler configurations. When beta=0.0 (i.e., the KL term is ignored), it is
    equivalent to a vanilla autoencoder.

    Additionally, this implementation supports conditional variational autoencoders (cVAE), enabling the model to learn and
    generate data conditioned on auxiliary inputs such as class labels, process states, or other side information. Conditioning
    vectors can be provided during training, inference, and sampling.
    """

    def __init__(
        self,
        encoder_shape=(256, 128),
        latent_dim=32,
        decoder_shape=None,
        beta=1.0,
        kl_warmup_epochs=0,
        kl_annealing_epochs=100,
        free_bits_threshold=0.0,
        activation_function="relu",
        optimizer="adam",
        scheduler=None,
        batch_norm_momentum=None,
        dropout_ratio=0.0,
        progressive_dropout=False,
        n_epochs=100,
        early_stopping=False,
        patience=10,
        min_delta=0.0,
        batch_size=None,
        seed=None,
        **kwargs,
    ):
        if decoder_shape is None:
            decoder_shape = tuple(reversed(encoder_shape))

        # Minimal inlined "base model" scaffolding: a hyperparameters dict
        # populated from every argument passed here (mirrors what
        # BaseModel.__init__ used to do via super().__init__(**kwargs)),
        # plus is_fit / is_classification / input_example bookkeeping.
        self.hyperparameters = dict(
            encoder_shape=encoder_shape,
            latent_dim=latent_dim,
            decoder_shape=decoder_shape,
            beta=beta,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_annealing_epochs=kl_annealing_epochs,
            free_bits_threshold=free_bits_threshold,
            activation_function=activation_function,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_norm_momentum=batch_norm_momentum,
            dropout_ratio=dropout_ratio,
            progressive_dropout=progressive_dropout,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        self.is_fit = False
        self.is_classification = False
        self.input_example = None

        self.device = get_best_device()

        self.n_input = None
        self.n_output = None
        self.cond_dim = None
        self.encoder = None
        self.decoder = None
        self.latent_dim = latent_dim

        self.optimizer = None
        self.scheduler = None
        self.n_net_parameters = None

        self.cur_epoch = 0

        self.is_autoencoder = self.hyperparameters["beta"] == 0.0
        self.deterministic = self.is_autoencoder

        # get activation function and loss function
        activation_func_class, activations_func_kwargs = get_activation_validate_kwargs(
            self.hyperparameters["activation_function"], **kwargs
        )
        self.activation_function = activation_func_class(**activations_func_kwargs)

        loss_func_class, loss_func_kwargs = get_loss_validate_kwargs("mse")
        self.loss_function = loss_func_class(**loss_func_kwargs)

        # get optimizer and scheduler classes as well as their kwargs
        self.optimizer_class, self.optimizer_kwargs = get_optimizer_validate_kwargs(
            self.hyperparameters["optimizer"], **kwargs
        )

        self.scheduler_class, self.scheduler_kwargs = get_scheduler_validate_kwargs(
            self.hyperparameters["scheduler"], **kwargs
        )

        # create early stopper
        self.early_stopper = None
        if early_stopping:
            self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        # create progressive dropout instance
        if self.hyperparameters["progressive_dropout"]:
            self.progressive_dropout = ProgressiveDropout(dim=-1, keep=-1, renorm=True)
        else:
            self.progressive_dropout = None

        # set seed
        if seed is not None:
            set_all_seeds(seed)

    # -- inlined base-model scaffolding (was BaseModel/BaseTabularModel/
    #    BaseGenerativeModel) -------------------------------------------

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters.update(kwargs)

    def fit(
        self,
        X_train,
        y_train,
        c_train=None,
        X_validation=None,
        y_validation=None,
        c_validation=None,
    ):
        losses = self._fit(
            X_train,
            y_train,
            c_train=c_train,
            X_validation=X_validation,
            y_validation=y_validation,
            c_validation=c_validation,
        )
        self.is_fit = True
        self.input_example = X_train
        return losses

    def predict(self, X, c=None):
        if not self.is_fit:
            raise RuntimeError("Model must be fit before calling predict().")
        return self._predict(X, c=c)

    def prior(self, num_samples, **kwargs):
        return np.random.randn(num_samples, self.hyperparameters["latent_dim"])

    def sample(self, c=None, z=None, num_samples=None, return_latents=False, **kwargs):
        if z is not None:
            if c is not None and len(c) != len(z):
                raise ValueError(
                    f"Batch size mismatch: c has {len(c)} samples but z has {len(z)}."
                )
        elif c is not None:
            z = self.prior(num_samples=len(c))
        else:
            if num_samples is None:
                raise ValueError(
                    "num_samples must be specified for unconditional sampling (when both c and z are None)."
                )
            z = self.prior(num_samples=num_samples)
        return self._sample(z, c, return_latents=return_latents, **kwargs)

    # -- ported VAE internals -------------------------------------------

    @staticmethod
    def __unsqueeze(data):
        if len(data.shape) == 1:
            return data.reshape(-1, 1)
        else:
            return data

    @staticmethod
    def __squeeze(data):
        if len(data.shape) == 2 and data.shape[1] == 1:
            return data.squeeze()
        else:
            return data

    def __create_model(self, n_input, cond_dim=0):
        self.encoder = Encoder(
            n_input=n_input,
            hidden_layers=self.hyperparameters["encoder_shape"],
            latent_dim=self.hyperparameters["latent_dim"],
            cond_dim=cond_dim,
            activation_function=self.activation_function,
            batch_norm_momentum=self.hyperparameters["batch_norm_momentum"],
            dropout_ratio=self.hyperparameters["dropout_ratio"],
        ).to(self.device)

        self.decoder = Decoder(
            latent_dim=self.hyperparameters["latent_dim"],
            hidden_layers=self.hyperparameters["decoder_shape"],
            output_dim=n_input,
            cond_dim=cond_dim,
            activation_function=self.activation_function,
            batch_norm_momentum=self.hyperparameters["batch_norm_momentum"],
            dropout_ratio=self.hyperparameters["dropout_ratio"],
        ).to(self.device)

        self.n_net_parameters = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        ) + sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def _reparameterize(self, mu, logvar):
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32).to(self.device)
        if not isinstance(logvar, torch.Tensor):
            logvar = torch.tensor(logvar, dtype=torch.float32).to(self.device)

        if self.deterministic:
            return mu

        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _forward_pass(self, X, c=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        mu, logvar = self.encoder(X, c)
        z = self._reparameterize(mu, logvar)

        # Apply progressive dropout only during training
        if self.encoder.training and self.progressive_dropout is not None:
            z = self.progressive_dropout(z)

        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar

    def make_deterministic(self):
        if not self.is_fit:
            raise RuntimeError("VAE model must be fit before making it deterministic.")
        self.deterministic = True

    def make_stochastic(self):
        if self.is_autoencoder:
            raise RuntimeError(
                "Model must be a VAE to make it stochastic. i.e., the model is initialized with beta > 0.0"
            )
        self.deterministic = False

    def __vae_loss(self, x_recon, x_true, mu, logvar):
        recon_loss = self.loss_function(x_recon, x_true)  # e.g., MSE
        kl_per_dim = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )  # shape: [batch_size, latent_dim]

        # Free bits trick: limit the minimum contribution of each dimension
        if self.hyperparameters["free_bits_threshold"] > 0.0:
            kl_per_dim = torch.maximum(
                kl_per_dim,
                self.hyperparameters["free_bits_threshold"]
                * torch.ones_like(kl_per_dim),
            )

        # Sum over latent dimensions, then mean over batch
        kl_div = torch.mean(torch.sum(kl_per_dim, dim=1))

        # KL warmup and annealing
        warmup_epochs = self.hyperparameters["kl_warmup_epochs"]
        anneal_epochs = self.hyperparameters["kl_annealing_epochs"]
        max_beta = self.hyperparameters["beta"]

        epoch_count = self.cur_epoch - warmup_epochs

        if epoch_count < 0:
            beta = 0.0
        else:
            beta = (
                min(epoch_count / anneal_epochs, 1.0) * max_beta
                if anneal_epochs > 0
                else max_beta
            )

        return recon_loss + beta * kl_div, recon_loss, beta * kl_div

    def _fit(
        self,
        X_train,
        y_train,
        c_train=None,
        X_validation=None,
        y_validation=None,
        c_validation=None,
    ):
        X_train = self.__unsqueeze(X_train)
        y_train = self.__unsqueeze(y_train)
        if X_validation is not None and y_validation is not None:
            X_validation = self.__unsqueeze(X_validation)
            y_validation = self.__unsqueeze(y_validation)

        if c_train is not None:
            c_train = self.__unsqueeze(c_train)
            if X_validation is not None and c_validation is not None:
                c_validation = self.__unsqueeze(c_validation)

        # create data loaders
        batch_size = self.hyperparameters["batch_size"]
        if batch_size is None:
            batch_size = get_default_batch_size(X_train.shape[0])
            self.hyperparameters["batch_size"] = batch_size

        train_loader = create_data_loader(
            X=X_train,
            y=y_train,
            c=c_train,
            batch_size=batch_size,
            seed=self.hyperparameters["seed"],
        )
        val_loader = None
        if X_validation is not None and y_validation is not None:
            val_loader = create_data_loader(
                X=X_validation,
                y=y_validation,
                c=c_validation,
                batch_size=batch_size,
                drop_last=False,
                seed=self.hyperparameters["seed"],
            )

        if not self.is_fit:
            # create model
            self.n_input = X_train.shape[-1]
            self.n_output = y_train.shape[-1]
            self.cond_dim = c_train.shape[-1] if c_train is not None else 0
            self.__create_model(self.n_input, self.cond_dim)

            # create optimizer and scheduler
            self.optimizer = self.optimizer_class(
                itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                **self.optimizer_kwargs,
            )

            if self.scheduler_class is not None:
                self.scheduler = self.scheduler_class(
                    self.optimizer, **self.scheduler_kwargs
                )

            self.losses = {"train": [], "train_recon": [], "train_kl": []}
            if val_loader is not None:
                self.losses["val"] = []
                self.losses["val_recon"] = []
                self.losses["val_kl"] = []

        if not self.is_autoencoder:
            self.make_stochastic()

        # Train the model
        with tqdm(
            range(self.hyperparameters["n_epochs"]),
            desc="Training Progress",
            dynamic_ncols=True,
        ) as pbar:
            for _ in pbar:
                # Train the net
                train_loss, train_recon_loss, train_kl_div_loss = self.__train_epoch(
                    train_loader
                )
                self.losses["train"].append(train_loss)
                self.losses["train_recon"].append(train_recon_loss)
                self.losses["train_kl"].append(train_kl_div_loss)

                # Evaluate the net
                validation_loss = None
                if val_loader is not None:
                    validation_loss, validation_recon_loss, validation_kl_div_loss = (
                        self.__evaluate_epoch(val_loader)
                    )
                    self.losses["val"].append(validation_loss)
                    self.losses["val_recon"].append(validation_recon_loss)
                    self.losses["val_kl"].append(validation_kl_div_loss)

                # Update progress bar with loss values (validation only shows if it is being done)
                pbar.set_postfix(**{k: v[-1] for k, v in self.losses.items()})

                # early stopping
                if validation_loss is not None and self.early_stopper:
                    if self.early_stopper(validation_loss):
                        self.hyperparameters["best_epoch"] = self.cur_epoch
                        break
                if (
                    "best_epoch" in self.hyperparameters
                    and self.cur_epoch >= self.hyperparameters["best_epoch"]
                ):
                    break

                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # update epoch count
                self.cur_epoch += 1

        return self.losses

    def __train_epoch(self, data_loader):
        self.encoder.train()
        self.decoder.train()

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_div_loss = 0
        for batch in data_loader:
            if len(batch) == 2:
                X, y = batch
                c = None
            else:
                X, y, c = batch
            X = X.to(self.device)
            y = y.to(self.device)
            if c is not None:
                c = c.to(self.device)

            # zero gradients
            self.optimizer.zero_grad()

            # make prediction
            x_recon, mu, logvar = self._forward_pass(X, c)

            # calculate loss
            loss, recon_loss, kl_div_loss = self.__vae_loss(x_recon, X, mu, logvar)
            epoch_loss += loss.data.cpu().detach().numpy() * X.shape[0]
            epoch_recon_loss += recon_loss.data.cpu().detach().numpy() * X.shape[0]
            epoch_kl_div_loss += kl_div_loss.data.cpu().detach().numpy() * X.shape[0]

            # back propagation
            loss.backward()
            self.optimizer.step()

        epoch_loss /= len(data_loader.dataset)
        epoch_recon_loss /= len(data_loader.dataset)
        epoch_kl_div_loss /= len(data_loader.dataset)

        return epoch_loss, epoch_recon_loss, epoch_kl_div_loss

    def __evaluate_epoch(self, data_loader):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_div_loss = 0

            for batch in data_loader:
                if len(batch) == 2:
                    X, y = batch
                    c = None
                else:
                    X, y, c = batch
                X = X.to(self.device)
                y = y.to(self.device)
                if c is not None:
                    c = c.to(self.device)

                # make prediction
                x_recon, mu, logvar = self._forward_pass(X, c)

                # calculate loss
                loss, recon_loss, kl_div_loss = self.__vae_loss(x_recon, X, mu, logvar)
                epoch_loss += loss.data.cpu().detach().numpy() * X.shape[0]
                epoch_recon_loss += recon_loss.data.cpu().detach().numpy() * X.shape[0]
                epoch_kl_div_loss += (
                    kl_div_loss.data.cpu().detach().numpy() * X.shape[0]
                )

        epoch_loss /= len(data_loader.dataset)
        epoch_recon_loss /= len(data_loader.dataset)
        epoch_kl_div_loss /= len(data_loader.dataset)

        return epoch_loss, epoch_recon_loss, epoch_kl_div_loss

    def encode(self, X, c=None):
        X = self.__unsqueeze(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if c is not None:
            c = self.__unsqueeze(c)
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        self.encoder.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(X_tensor, c)

        return mu.cpu().detach().numpy(), logvar.cpu().detach().numpy()

    def decode(self, z, c=None):
        z = self.__unsqueeze(z)
        z = torch.tensor(z, dtype=torch.float32).to(self.device)

        if c is not None:
            c = self.__unsqueeze(c)
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        self.decoder.eval()
        with torch.no_grad():
            x_recon = self.decoder(z, c)

        return x_recon.cpu().detach().numpy()

    def _predict(self, X, c=None):
        X = self.__unsqueeze(X)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if c is not None:
            c = self.__unsqueeze(c)
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            x_recon, _, _ = self._forward_pass(X, c=c)

        x_recon = x_recon.cpu().detach().numpy()

        return x_recon

    def _sample(self, z, c, return_latents=False, **kwargs):
        z = self.__unsqueeze(z)

        if c is not None:
            c = self.__unsqueeze(c)

        x_recon = self.decode(z, c)

        if return_latents:
            return x_recon, z
        return x_recon

    def score(self, X, X_recon, metric="mse"):
        X = np.asarray(X)
        X_recon = np.asarray(X_recon)
        if metric == "mse":
            return float(np.mean((X - X_recon) ** 2))
        elif metric == "mae":
            return float(np.mean(np.abs(X - X_recon)))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def evaluate_progressive_dropout_sweep(self, X, keep_values, metric="mse", c=None):
        if self.progressive_dropout is None:
            raise RuntimeError(
                "Progressive dropout must be enabled to run this evaluation. Retrain with progressive_dropout = True."
            )

        # Save original keep value
        original_keep = self.progressive_dropout.keep

        # Make sure model is in eval mode (no training dropout)
        self.encoder.eval()
        self.decoder.eval()

        results = {}
        for k in keep_values:
            self.progressive_dropout.keep = k

            with torch.no_grad():
                mu, logvar = self.encode(X, c)
                z = self._reparameterize(mu, logvar)
                z = self.progressive_dropout(z)
                X_recon = self.decode(z, c)

            results[k] = self.score(X, X_recon, metric=metric)

        # Restore original keep value
        self.progressive_dropout.keep = original_keep

        return results

    # setters and getters for n_epoch
    def get_n_epochs(self):
        return self.hyperparameters["n_epochs"]

    def set_n_epochs(self, n_epochs):
        if isinstance(n_epochs, int) and n_epochs > 0:
            self.set_hyperparameters(n_epochs=n_epochs)
        else:
            raise ValueError("n_epochs must be a positive integer.")


class Encoder(torch.nn.Module):
    def __init__(
        self,
        n_input,
        hidden_layers,
        latent_dim,
        activation_function,
        batch_norm_momentum=None,
        dropout_ratio=0.0,
        cond_dim=0,
    ):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.batch_norms = (
            torch.nn.ModuleList() if batch_norm_momentum is not None else None
        )
        self.activation_function = activation_function

        channels = n_input + cond_dim
        for size in hidden_layers:
            self.hidden_layers.append(torch.nn.Linear(channels, size))
            if self.batch_norms is not None:
                self.batch_norms.append(
                    torch.nn.BatchNorm1d(size, momentum=batch_norm_momentum)
                )
            self.dropouts.append(torch.nn.Dropout(dropout_ratio))
            channels = size

        # output layers for latent mean and log variance
        self.mu_layer = torch.nn.Linear(channels, latent_dim)
        self.logvar_layer = torch.nn.Linear(channels, latent_dim)

    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=-1)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self.activation_function(x)
            x = self.dropouts[i](x)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar


class Decoder(torch.nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_layers,
        output_dim,
        activation_function,
        batch_norm_momentum=None,
        dropout_ratio=0.0,
        cond_dim=0,
    ):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.batch_norms = (
            torch.nn.ModuleList() if batch_norm_momentum is not None else None
        )
        self.activation_function = activation_function

        channels = latent_dim + cond_dim
        for size in hidden_layers:
            self.hidden_layers.append(torch.nn.Linear(channels, size))
            if self.batch_norms is not None:
                self.batch_norms.append(
                    torch.nn.BatchNorm1d(size, momentum=batch_norm_momentum)
                )
            self.dropouts.append(torch.nn.Dropout(dropout_ratio))
            channels = size

        self.output_layer = torch.nn.Linear(channels, output_dim)

    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=-1)

        for i, layer in enumerate(self.hidden_layers):
            z = layer(z)
            if self.batch_norms is not None:
                z = self.batch_norms[i](z)
            z = self.activation_function(z)
            z = self.dropouts[i](z)

        x_recon = self.output_layer(z)
        return x_recon


class ProgressiveDropout(torch.nn.Module):
    """
    A custom PyTorch module for progressively dropping dimensions from a tensor.

    This is useful for VAEs, where the latent space should be progressively reduced
    as training progresses. This module will randomly drop a subset of the dimensions
    from the input tensor, and optionally scale the remaining activations to
    preserve the magnitude of the tensor.
    """

    def __init__(self, dim=-1, keep=-1, renorm=True):
        super().__init__()
        self.dim = dim
        self.keep = keep
        self.renorm = renorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.keep == -1:
            return x
        mask = self._generate_mask(x)
        return x * mask

    def _generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            keep_indices = self._get_keep_indices(x)  # shape: (batch_size,)
            dim_size = x.size(self.dim)  # usually latent_dim
            batch_size = x.size(0)

            # Binary mask of shape (batch_size, dim_size)
            mask = torch.arange(dim_size, device=x.device).unsqueeze(
                0
            ) < keep_indices.unsqueeze(1)
            mask = mask.to(x.dtype)  # shape: (batch_size, latent_dim)

            if self.renorm and not torch.all(keep_indices == 0):
                norm_factor = dim_size / keep_indices.clamp(min=1)
                mask *= norm_factor.unsqueeze(1)

            # Reshape mask to match x
            target_shape = [1] * x.dim()
            target_shape[0] = batch_size
            target_shape[self.dim] = dim_size
            mask = mask.view(*target_shape)

            return mask

    def _get_keep_indices(self, x: torch.Tensor) -> torch.Tensor:
        latent_dim = x.size(self.dim)
        if self.keep == -1:
            return torch.randint(1, latent_dim + 1, (x.size(0),), device=x.device)
        else:
            return torch.full((x.size(0),), self.keep, device=x.device)
