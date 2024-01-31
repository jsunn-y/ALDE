from __future__ import annotations

from collections.abc import Sequence, Mapping
from dataclasses import dataclass, astuple
from src.utils import MapClass

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
import gpytorch
import gpytorch.distributions as gdist
import numpy as np
import torch
from torch import Tensor

class GenericModel:
    #general model for GP_BOTORCH, DKL_BOTORCH, and DNN_ENSEMBLE
    def __init(self, **_):
        raise NotImplementedError

    def train(
        self,
    ):
        raise NotImplementedError

    def predict(
        self,
    ):
        raise NotImplementedError

    def embedding(
        self,
    ):  # optional
        raise NotImplementedError

    def forward(
        self,
    ):
        """Not used since all training occurs in train()."""
        raise NotImplementedError

    def get_kernel_noise(
        self,
    ): # only if needed for thompson sampling
        raise NotImplementedError


# other: training data, base (GP) kernel, architecture, likelihood
@dataclass
class NET_ARGS(MapClass):
    """General model class args. Each model class can
    take some subset of these as arguments; others discarded."""

    train_x: Tensor = None  
    train_y: Tensor = None
    architecture: Sequence[
        int
    ] = None
    activation: str = "relu"
    kernel: str = "lin"
    p_dropout: float = 0.0
    likelihood: gpytorch.likelihoods.Likelihood = None
    device: torch.device | str = "cuda"
    inference_args: OPT_ARGS | SAMP_ARGS = None  # use one of below!

@dataclass
class OPT_ARGS(MapClass):
    """Struct for models that use a standard optimization protocol (GP, DKL, SVI).
    Intentionally leaving out: optimizer (Adam), xxx."""

    lr: float = 1e-2
    num_iter: int = 100
    verbose: int = 1


@dataclass
class SAMP_ARGS(MapClass):
    """Struct for models that use a sampling-based protocol (MCMC).
    Intentionally leaving out: kernel (NUTS), init_strategy (median)."""

    num_warmup: int = 100
    num_samples: int = 100
    num_chains: int = 1
    thinning: int = 2
    verbose: int = 1


class DNN_FF(torch.nn.Sequential):
    """
    Standard DNN, feedforward.
    For use in DKL and DNN_ENSEMBLE
    """
    act_dict = {
        "relu": torch.nn.ReLU(),
        "lrelu": torch.nn.LeakyReLU(),
        "swish": torch.nn.SiLU(),
        "sigmoid": torch.nn.Sigmoid(),
        "tanh": torch.nn.Tanh(),
        "softmax": torch.nn.Softmax(),
    }

    def __init__(
        self,
        architecture,
        activation="relu",
        p_dropout=0,
        inference_args=None,
        device='cuda',
        *_,
        **__,
    ):
        super().__init__()
        self.device = device
        self.architecture = architecture
        act_layer = self.act_dict[activation.lower()]
        self.inference_args = inference_args
        self.dkl = True

        #print(architecture)
        for dim in range(len(architecture)):
            name = str(dim + 1)
            if dim + 1 < len(architecture):
                self.add_module(
                    "linear" + name,
                    torch.nn.Linear(architecture[dim], architecture[dim + 1]).double(),
                )
                # don't dropout from output layer ie add below
            if dim + 2 < len(architecture):
                if p_dropout > 0 and p_dropout < 1:
                    self.add_module("dropout" + name, torch.nn.Dropout(p=p_dropout))
                name = activation + name
                self.add_module(name, act_layer)

    def get_params(self):
        return [{"params": self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        self.train()
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        mse = torch.nn.MSELoss()
        losses = np.zeros(num_iter)
        w = 30  # moving window size for early stopping

        for i in range(num_iter):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = mse(preds, Y)
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()

            if i > w:
                recent_min = losses[i-w+1:i+1].min() 
                overall_min = losses[:i-w+1].min()
                if overall_min <= recent_min:
                    print("Early stopping at iteration " + str(i))
                    break

        self.eval()
        return self, None

class BoTorchGP(SingleTaskGP, GenericModel):
    """
    GP or DKL model.
    """
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        kernel,
        architecture,
        activation=None,
        p_dropout=0,
        inference_args=None,
        device='cuda',
	    use_own_default_likelihood=False,
        *_,
        **__,
    ):
        """
        Init GP or DKL.

        """
        self.dkl = False
        self.device = device
        self.gpu_batch_size = 1000
        
        self.architecture = architecture
        
        self.lin = False
        self.feature_extractor = None

        if len(architecture) > 2:  # DKL. could also allow this to be pretrained and passed in
            self.dkl = True
            # chop GPR part of arc off, from E --> 1
            if len(architecture) >= 6: #CNN
                self.cdkl = True

        if kernel is None or kernel.lower() == "rbf":
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    has_lengthscale=True,
                    ard_num_dims=architecture[-2],
                    num_dims=architecture[-2],
                )
            )
        # elif kernel.lower() in ["lin", "linear",]:
        #     covar_module = gpytorch.kernels.LinearKernel()
        #     self.lin = True
        else:
            raise NotImplementedError(f"Add your kernel {kernel} in networks.py to use it.")

        SingleTaskGP.__init__(
            self,
            train_X=train_x.double(),
            train_Y=train_y.unsqueeze(-1).double(),
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
        
        if not use_own_default_likelihood:
            self.likelihood = likelihood

        if self.dkl:
            self.feature_extractor = DNN_FF(architecture[:-1], activation, p_dropout)
        
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        emb = self.embedding(x) #pass encodings through the neural network (feature extractor)
        emb = self.scale_to_bounds(emb)
        
        mean_x = self.mean_module(emb)
        covar_x = self.covar_module(emb)
        return gdist.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x: Tensor) -> Tensor:
        if self.dkl:
            return self.feature_extractor(x)
        else:
            return x

    def get_params(self):
        if self.dkl:
            return self.feature_extractor.get_params() + [
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ]
        else:
            return [{"params": self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, max_iter=300, *_, **__):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        losses = np.zeros(num_iter)
        w = 30  # moving window size for early stopping
        

        if self.feature_extractor != None:
            self.feature_extractor.train()
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
            for i in range(num_iter):
                optimizer.zero_grad()
                preds = self(X)
                loss = -mll(preds, Y)
                loss.backward()
                optimizer.step()
                losses[i] = loss.item()

                if i > w:
                    recent_min = losses[i-w+1:i+1].min() 
                    overall_min = losses[:i-w+1].min()
                    if overall_min <= recent_min:
                        print("Early stopping at iteration " + str(i))
                        break
        if self.feature_extractor != None:
            self.feature_extractor.eval()
            
        self.eval()
        self.likelihood.eval()
        return None

    def predict_batched_gpu(self, X):
        mu, sigma = [], []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            mvn = self(X[n : n + self.gpu_batch_size].to(self.device))
            mu.append(mvn.mean.cpu())
            sigma.append(mvn.stddev.cpu())
        return torch.cat(mu, 0), torch.cat(sigma, 0)

    def embed_batched_gpu(self, X):
        emb = torch.zeros((X.shape[0], self.architecture[-2]))
        for n in range(0, X.shape[0], self.gpu_batch_size):
            emb[n:n+self.gpu_batch_size, :] = self.embedding(X[n : n + self.gpu_batch_size].to(self.device)).detach()
        return emb

    def eval_acquisition_batched_gpu(self, X, f=(lambda x: x)):
        acq = []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            acq.append(f(X[n : n + self.gpu_batch_size].to(self.device)).detach())
        return torch.cat(acq, 0)

    def get_kernel_noise(self):
        return self.likelihood.noise.cpu().detach()