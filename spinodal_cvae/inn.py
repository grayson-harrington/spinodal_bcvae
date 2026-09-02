# external imports
import numpy as np
import torch
from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# Standalone torch/stdlib helpers, shared with vae.py. These replace what the
# original private lab package obtained from its shared model-utilities module.
from ._torch_utils import (
    EarlyStopping,
    create_data_loader,
    get_activation_validate_kwargs,
    get_best_device,
    get_default_batch_size,
    get_loss_validate_kwargs,
    get_optimizer_validate_kwargs,
    get_scheduler_validate_kwargs,
    set_all_seeds,
)


# ---------------------------------------------------------------------------
# InvertibleNeuralNetwork (conditional normalizing flow) and its SubNet.
#
# Ported from the original private lab package's generative INN module. The
# FrEIA computation graph, the standard-normal-prior NLL objective, and the
# training loop are kept verbatim. What changed: the class no longer inherits
# from BaseGenerativeModel -> BaseTabularModel -> BaseModel; instead a minimal
# self-contained base (hyperparameters dict, is_fit flag, thin fit/predict/
# sample wrappers) is inlined below, and calls to now-removed private-package
# utility functions are replaced with the local equivalents in _torch_utils.
# The FrEIA dependency is kept (public PyPI package) rather than reimplemented.
# ---------------------------------------------------------------------------


class InvertibleNeuralNetwork:
    """
    Invertible Neural Network (INN) implementation.

    This class implements an INN using the FrEIA library. It is a bijective
    generative model: it can generate samples and compute the exact likelihood
    of existing samples. The architecture is a sequence of ``n_blocks``
    ``AllInOneBlock`` coupling transformations (affine coupling + soft clamp +
    ActNorm-style global affine + permutation). Training minimizes the
    negative log-likelihood under a standard-normal prior with an Adam
    optimizer and an optional learning-rate schedule.
    """

    def __init__(
        self,
        # --- INN architecture hyperparameters ---
        n_blocks=6,
        block_hidden_shape=128,
        activation_function="relu",
        # --- AllInOne coupling block params ---
        clamp=2.0,
        gin_block=False,
        global_affine_init=1.0,
        global_affine_type="SOFTPLUS",
        permute_soft=False,
        learned_householder_permutation=0,
        reverse_permutation=False,
        # --- Training hyperparameters ---
        optimizer="adam",
        scheduler=None,
        n_epochs=100,
        batch_size=None,
        early_stopping=False,
        patience=10,
        min_delta=0.0,
        seed=None,
        **kwargs,
    ):
        latent_dim = kwargs.pop("latent_dim", "set_later")

        # Minimal inlined "base model" scaffolding: a hyperparameters dict
        # populated from every argument passed here (mirrors what
        # BaseModel.__init__ used to do via super().__init__(**kwargs)),
        # plus is_fit / is_classification / input_example bookkeeping.
        self.hyperparameters = dict(
            n_blocks=n_blocks,
            block_hidden_shape=block_hidden_shape,
            activation_function=activation_function,
            clamp=clamp,
            gin_block=gin_block,
            global_affine_init=global_affine_init,
            global_affine_type=global_affine_type,
            permute_soft=permute_soft,
            learned_householder_permutation=learned_householder_permutation,
            reverse_permutation=reverse_permutation,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            latent_dim=latent_dim,
            **kwargs,
        )
        self.is_fit = False
        self.is_classification = False
        self.input_example = None

        self.device = get_best_device()

        self.optimizer = None
        self.scheduler = None
        self.n_net_parameters = None

        self.cur_epoch = 0

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

        # create subnet_constructor
        self.subnet_constructor = self.subnet_constructor_constructor(
            self.hyperparameters["block_hidden_shape"],
            self.activation_function,
        )

        # create early stopper
        self.early_stopper = None
        if early_stopping:
            self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

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
                    "num_samples must be specified for unconditional sampling "
                    "(when both c and z are None)."
                )
            z = self.prior(num_samples=num_samples)
        return self._sample(z, c, return_latents=return_latents, **kwargs)

    # -- ported INN internals -----------------------------------------------

    @staticmethod
    def subnet_constructor_constructor(h_dim, activation_function):
        def subnet_constructor(in_dim, out_dim):
            return SubNet(
                in_dim,
                h_dim if isinstance(h_dim, tuple) else (h_dim,),
                out_dim,
                activation_function,
            )

        return subnet_constructor

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

    def __create_model(self, input_dim, cond_dim=0):
        block_kwargs = {
            "subnet_constructor": self.subnet_constructor,
            "affine_clamping": self.hyperparameters["clamp"],
            "gin_block": self.hyperparameters["gin_block"],
            "global_affine_init": self.hyperparameters["global_affine_init"],
            "global_affine_type": self.hyperparameters["global_affine_type"],
            "permute_soft": self.hyperparameters["permute_soft"],
            "learned_householder_permutation": self.hyperparameters[
                "learned_householder_permutation"
            ],
            "reverse_permutation": self.hyperparameters["reverse_permutation"],
        }

        inn = Ff.SequenceINN(input_dim)

        for _ in range(self.hyperparameters["n_blocks"]):
            inn.append(
                Fm.AllInOneBlock,
                cond=None if cond_dim == 0 else 0,
                cond_shape=None if cond_dim == 0 else (cond_dim,),
                **block_kwargs,
            )

        self.n_net_parameters = sum([np.prod(p.size()) for p in inn.parameters()])

        self.inn = inn.to(self.device)

    def _forward_pass(self, X, c=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        z, log_jac_det = self.inn(X, [c])

        return z, log_jac_det

    def _backward_pass(self, X, c=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        z, log_jac_det = self.inn(X, [c], rev=True)

        return z, log_jac_det

    def __inn_loss(self, z, log_jac_det):
        loss = 0.5 * torch.sum(z**2, dim=1) - log_jac_det
        loss = loss.mean() / z.shape[-1]

        return loss

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

        self.hyperparameters["latent_dim"] = X_train.shape[-1]

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
            self.cond_dim = c_train.shape[-1] if c_train is not None else 0

            # create model
            self.__create_model(
                self.hyperparameters["latent_dim"], cond_dim=self.cond_dim
            )

            # create optimizer and scheduler
            self.optimizer = self.optimizer_class(
                self.inn.parameters(),
                **self.optimizer_kwargs,
            )

            if self.scheduler_class is not None:
                self.scheduler = self.scheduler_class(
                    self.optimizer, **self.scheduler_kwargs
                )

        # Train the model
        self.losses = {"train": []}
        if val_loader is not None:
            self.losses["val"] = []

        with tqdm(
            range(self.hyperparameters["n_epochs"]),
            desc="Training Progress",
            dynamic_ncols=True,
        ) as pbar:
            for _ in pbar:
                # Train the net
                train_loss = self.__train_epoch(train_loader)
                self.losses["train"].append(train_loss)

                # Evaluate the net
                validation_loss = None
                if val_loader is not None:
                    validation_loss = self.__evaluate_epoch(val_loader)
                    self.losses["val"].append(validation_loss)

                # Update progress bar with loss values
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
        self.inn.train()

        epoch_loss = 0
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
            z, log_jac_det = self._forward_pass(X, c)

            # calculate loss
            loss = self.__inn_loss(z, log_jac_det)
            epoch_loss += loss.data.cpu().detach().numpy() * X.shape[0]

            # back propagation
            loss.backward()
            self.optimizer.step()

        epoch_loss /= len(data_loader.dataset)

        return epoch_loss

    def __evaluate_epoch(self, data_loader):
        self.inn.eval()
        with torch.no_grad():
            epoch_loss = 0

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
                z, log_jac_det = self._forward_pass(X, c)

                # calculate loss
                loss = self.__inn_loss(z, log_jac_det)
                epoch_loss += loss.data.cpu().detach().numpy() * X.shape[0]

        epoch_loss /= len(data_loader.dataset)

        return epoch_loss

    def encode(self, X, c=None):
        X = self.__unsqueeze(X)
        if c is not None:
            c = self.__unsqueeze(c)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        self.inn.eval()
        with torch.no_grad():
            z, log_jac_det = self._forward_pass(X, c)

        return z.cpu().detach().numpy(), log_jac_det.cpu().detach().numpy()

    def decode(self, z, c=None):
        z = self.__unsqueeze(z)
        if c is not None:
            c = self.__unsqueeze(c)

        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        self.inn.eval()
        with torch.no_grad():
            x_recon, _ = self._backward_pass(z, c)

        return x_recon.cpu().detach().numpy()

    def _predict(self, X, c=None):
        X = self.__unsqueeze(X)
        if c is not None:
            c = self.__unsqueeze(c)

        self.inn.eval()
        with torch.no_grad():
            z, _ = self.encode(X, c)
            x_recon = self.decode(z, c)

        return x_recon

    def _sample(self, z, c, return_latents=False, **kwargs):
        z = self.__unsqueeze(z)
        if c is not None:
            c = self.__unsqueeze(c)

        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32).to(self.device)

        x_recon = self.decode(z, c)

        if return_latents:
            return x_recon, z
        return x_recon

    def _get_state(self):
        return {
            "hyperparameters": self.hyperparameters,
            "inn_state_dict": self.inn.state_dict(),
            "optimizer_state_dict": (
                None if not self.optimizer else self.optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                None if not self.scheduler else self.scheduler.state_dict()
            ),
            "is_fit": self.is_fit,
            "cur_epoch": self.cur_epoch,
            "losses": self.losses,
            "cond_dim": getattr(self, "cond_dim", 0),
        }

    def _set_state(self, state):
        # restore stored metadata
        self.hyperparameters = state["hyperparameters"]
        self.is_fit = state.get("is_fit", False)
        self.cur_epoch = state.get("cur_epoch", 0)
        self.losses = state.get("losses", {})
        self.cond_dim = state.get("cond_dim", 0)

        # restore device
        self.device = get_best_device()

        # restore activation function
        activation_cls, activation_kwargs = get_activation_validate_kwargs(
            self.hyperparameters["activation_function"]
        )
        self.activation_function = activation_cls(**activation_kwargs)

        # restore loss function
        loss_cls, loss_kwargs = get_loss_validate_kwargs("mse")
        self.loss_function = loss_cls(**loss_kwargs)

        # restore optimizer configuration
        self.optimizer_class, self.optimizer_kwargs = get_optimizer_validate_kwargs(
            self.hyperparameters["optimizer"]
        )

        # restore scheduler configuration
        self.scheduler_class, self.scheduler_kwargs = get_scheduler_validate_kwargs(
            self.hyperparameters["scheduler"]
        )

        # restore subnet constructor
        self.subnet_constructor = self.subnet_constructor_constructor(
            self.hyperparameters["block_hidden_shape"],
            self.activation_function,
        )

        # restore early stopping configuration
        self.early_stopper = None
        if self.hyperparameters.get("early_stopping", False):
            self.early_stopper = EarlyStopping(
                patience=self.hyperparameters["patience"],
                min_delta=self.hyperparameters["min_delta"],
            )

        # rebuild INN graph
        self.__create_model(
            input_dim=self.hyperparameters["latent_dim"],
            cond_dim=self.cond_dim,
        )
        self.inn.to(self.device)

        # restore INN weights
        self.inn.load_state_dict(state["inn_state_dict"])

        # restore optimizer instance + state
        if state.get("optimizer_state_dict") is not None:
            self.optimizer = self.optimizer_class(
                self.inn.parameters(), **self.optimizer_kwargs
            )
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            self.optimizer = None

        # restore scheduler instance + state
        if (
            state.get("scheduler_state_dict") is not None
            and self.scheduler_class is not None
            and self.optimizer is not None
        ):
            self.scheduler = self.scheduler_class(
                self.optimizer, **self.scheduler_kwargs
            )
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        else:
            self.scheduler = None


class SubNet(torch.nn.Module):
    """
    A helper class for creating sub-networks used in the Invertible Neural
    Network (INN): a plain MLP with ``len(hidden_shape)`` hidden layers and a
    linear output head.
    """

    def __init__(
        self,
        n_input,
        hidden_shape,
        n_output,
        activation_function,
    ):
        super(SubNet, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        self.activation_function = activation_function

        nchannels = n_input
        for i in range(len(hidden_shape)):
            lay = torch.nn.Linear(nchannels, hidden_shape[i])
            self.hidden_layers.append(lay)
            nchannels = hidden_shape[i]
        self.predict = torch.nn.Linear(nchannels, n_output)

        # number of parameters in model
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.activation_function(x)
        x = self.predict(x)
        return x
