from __future__ import annotations

from collections.abc import Sequence, Mapping
from datetime import datetime
import time, os, sys
from typing import Literal
from dataclasses import dataclass, astuple
from src.utils import MapClass

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import gpytorch
import gpytorch.distributions as gdist
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import numpy as np
import torch
from torch import nn, Tensor

class GenericModel:
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


# could add priors to these, but will let users just mutate
# that in network classes b/c a lot, flexible


# other: training data, base (GP) kernel, architecture, likelihood
@dataclass
class NET_ARGS(MapClass):
    """General model class args. Each model class can
    take some subset of these as arguments; others discarded."""

    train_x: Tensor = None  # not always needed, can be discarded downstream
    train_y: Tensor = None  # "
    architecture: Sequence[
        int
    ] = None  # for GP, should be [inpdim, 1]; DNN [x,...,x,E,1]
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


#############################

# standard DNN, feedforward
class DNN_FF(torch.nn.Sequential):
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
        *_,
        **__,
    ):
        super().__init__()

        self.architecture = architecture
        act_layer = self.act_dict[activation.lower()]
        self.inference_args = inference_args
        self.dkl = True

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
        for iter in range(num_iter):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = mse(preds, Y)
            loss.backward()
            optimizer.step()

        self.eval()
        return self, None

#standard CNN
class CNN(torch.nn.Sequential):
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
        *_,
        **__,
    ):
        super().__init__()
        self.architecture = architecture #by default use [4, 20, 32, 32, 32, 64, 64] first is number of sites, second is the number of tokens (channels), middle layers are number of filters, last two are full connected
        act_layer = self.act_dict[activation.lower()]
        self.inference_args = inference_args
        self.dkl = True
        kernel_size = 2 #tests so far have been with 5
        n_sites = architecture[0]

        self.add_module("conv1d1", torch.nn.Conv1d(in_channels=architecture[1],
                                out_channels=architecture[2],
                                kernel_size=kernel_size,
                                padding='same').double(),)
        self.add_module(activation + '1', act_layer)
        self.add_module("conv1d2", torch.nn.Conv1d(in_channels=architecture[2],
                                out_channels=architecture[3],
                                kernel_size=kernel_size,
                                padding='same').double(),)
        self.add_module(activation + '2', act_layer)
        self.add_module("conv1d3", torch.nn.Conv1d(in_channels=architecture[3],
                                out_channels=architecture[4],
                                kernel_size=kernel_size,
                                padding='same').double(),)
        self.add_module(activation + '3', act_layer)
        self.add_module('Flatten', torch.nn.Flatten())
        self.add_module("linear1", torch.nn.Linear(architecture[4]*n_sites, architecture[5]).double(),)
        self.add_module(activation + '4', act_layer)
        if p_dropout > 0 and p_dropout < 1:
            self.add_module("dropout1", torch.nn.Dropout(p=p_dropout))
        self.add_module("linear2", torch.nn.Linear(architecture[5], architecture[6]).double(),)
        

    def get_params(self):
        return [{"params": self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        self.train()
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        mse = torch.nn.MSELoss()
        for iter in range(num_iter):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = mse(preds, Y)
            loss.backward()
            optimizer.step()

        self.eval()
        return self, None


class GP(gpytorch.models.ExactGP, GenericModel):
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
        *_,
        **__,
    ):
        """Init GP.
        -@param: training inputs (torch.tensor)
        -@param: training outputs (torch.tensor) corr. to inputs
        -@param: likelihood func(usually mll)
        -@param: outdim, depth of last layer of DNN
        -@param: a kernel (e.g. RBF, grid interp, spectral mixture, etc)
        -@param: grid_size, size of grid for grid interpolation
        -@param: aux variable (used for smoothness constant, etc.)
        """
        super().__init__(train_x, train_y, likelihood)
        self.dkl, self.cdkl = False, False
        self.device = device
        self.architecture = architecture
        self.gpu_batch_size = 1000

        # TODO: remove self.lin, idt necessary anymore?
        self.lin, self.feature_extractor = False, None
        self.inference_args = inference_args

        if (
            len(architecture) > 2
        ):  # DKL. could also allow this to be pretrained and passed in
            self.dkl = True
            # chop GPR part of arc off, from E --> 1
            if len(architecture) >= 6: #CNN
                self.cdkl = True
                self.feature_extractor = CNN(architecture[:-1], activation, p_dropout)
            else:
                self.feature_extractor = DNN_FF(architecture[:-1], activation, p_dropout)
                #not sure why this skipped the last one
                #self.feature_extractor = DNN_FF(architecture[:-1], activation, p_dropout)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == None or kernel.lower() == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    has_lengthscale=True,
                    ard_num_dims=architecture[-2],
                    num_dims=architecture[-2],
                )
            )
        elif kernel.lower() in [
            "lin",
            "linear",
        ]:  # TODO: should this be a scale kernel?
            self.covar_module = gpytorch.kernels.LinearKernel()
            self.lin = True
        else: raise NotImplementedError("Add your kernel in networks.py to use it.")

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        # We're first putting our data through a deep net (feature extractor)
        emb = self.embedding(x)
        mean_x = self.mean_module(emb)
        covar_x = self.covar_module(emb)
        return gdist.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x: Tensor) -> Tensor:
        # for use with TS acq
        if self.dkl:
            if self.cdkl:
                 #unflatten the array for CNN
                 n_sites = self.architecture[0]
                 n_tokens = int(x.shape[1]/n_sites)
                 
                 x = torch.transpose(torch.reshape(x, (x.shape[0], n_sites, n_tokens)), 1, 2)
                 
            return self.feature_extractor(x)
        else:
            return x

    def posterior(self, X=None, posterior_transform=None):
        # to conform to botorch model class
        self.eval()
        return self(X)

    def get_params(self):
        if self.dkl:
            return self.feature_extractor.get_params() + [
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ]
        else:
            return [{"params": self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        # TODO: add a verbose option?
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        #if not self.dkl:
        if False:
            self.likelihood, self = self.likelihood.cpu(), self.cpu()
            fit_gpytorch_mll(mll)
        else:
            if self.feature_extractor != None:
                self.feature_extractor.train()
            #adam is a first order optimization, LBFGSB is a better optimization but requires a hessian
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
                for iter in range(num_iter):
                    optimizer.zero_grad()
                    # TODO: shouldn't be forward
                    preds = self(X)
                    loss = -mll(preds, Y)
                    loss.backward()
                    #print("Loss: " + str(loss))
                    optimizer.step()

        if self.feature_extractor != None:
            self.feature_extractor.eval()
        self.eval()
        self.likelihood.eval()
        return None

    def get_kernel_noise(self):
        return self.likelihood.noise.cpu()

    def predict_batched_gpu(self, X):
        mu, sigma = [], []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            # TODO: forward gives prior, model uses posterior
            mvn = self(X[n : n + self.gpu_batch_size].to(self.device)).detach()
            mu.append(mvn.mean.cpu())
            sigma.append(mvn.stddev.cpu())
        return torch.cat(mu, 0), torch.cat(sigma, 0)

    def embed_batched_gpu(self, X):
        emb = torch.zeros((X.shape[0], self.architecture[-1]))
        for n in range(0, X.shape[0], self.gpu_batch_size):
            emb[n:n+self.gpu_batch_size, :] = self.embedding(X[n : n + self.gpu_batch_size].to(self.device)).detach()
        # print(emb[0].shape)
        return emb

    def eval_acquisition_batched_gpu(self, X, f=(lambda x: x)):
        acq = []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            acq.append(f(X[n : n + self.gpu_batch_size].to(self.device)).detach())
        # print(emb[0].shape)
        return torch.cat(acq, 0)

class BoTorchGP(SingleTaskGP, GenericModel):
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
        """Init GP.
        -@param: training inputs (torch.tensor)
        -@param: training outputs (torch.tensor) corr. to inputs
        -@param: likelihood func(usually mll)
        -@param: outdim, depth of last layer of DNN
        -@param: a kernel (e.g. RBF, grid interp, spectral mixture, etc)
        -@param: grid_size, size of grid for grid interpolation
        -@param: aux variable (used for smoothness constant, etc.)
        """
        self.dkl, self.cdkl = False, False
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
        elif kernel.lower() in ["lin", "linear",]:
            covar_module = gpytorch.kernels.LinearKernel()
            self.lin = True
        else:
            raise NotImplementedError(f"Add your kernel {kernel} in networks.py to use it.")

        SingleTaskGP.__init__(
            self,
            # TODO this doesn't work still bc training inputs diff
            train_X=train_x.double(),  # if not self.dkl else torch.zeros(train_x.shape[0], architecture[-2]).double(),
            train_Y=train_y.unsqueeze(-1).double(),
            covar_module=covar_module,
            # TODO: unclear what should be done here for NN outputs, etc.
            # input_transform=input_transform if not dkl else None,
            outcome_transform=Standardize(m=1),
        )
        
        if not use_own_default_likelihood:
            self.likelihood = likelihood

        if self.dkl:
            if self.cdkl:
                self.feature_extractor = CNN(architecture[:-1], activation, p_dropout)
            else:
                self.feature_extractor = DNN_FF(architecture[:-1], activation, p_dropout)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        # We're first putting our data through a deep net (feature extractor)
        emb = self.embedding(x)
        #something is wrong with the shape here in qEI
        
        mean_x = self.mean_module(emb)
        covar_x = self.covar_module(emb)
        return gdist.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x: Tensor) -> Tensor:
        # for use with TS acq
        if self.dkl:
            if self.cdkl:
                 #unflatten the array for CNN
                 n_sites = self.architecture[0]
                 n_tokens = int(x.shape[1]/n_sites)
                 x = torch.transpose(torch.reshape(x, (x.shape[0], n_sites, n_tokens)), 1, 2)
            
            # print(x.shape)     
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

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        if False:
        # if not self.dkl:
            self.likelihood, self = self.likelihood.cpu(), self.cpu()
            fit_gpytorch_mll(mll)
        else:
            if self.feature_extractor != None:
                self.feature_extractor.train()
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
                for iter in range(num_iter):
                    optimizer.zero_grad()
                    # IMPORTANT: don't use fwd
                    preds = self(X)
                    loss = -mll(preds, Y)
                    loss.backward()
                    #print("Loss: " + str(loss))
                    optimizer.step()
            if self.feature_extractor != None:
                self.feature_extractor.eval()

        self.eval()
        self.likelihood.eval()
        return None

    def predict_batched_gpu(self, X):
        mu, sigma = [], []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            # TODO: forward gives prior, model uses posterior
            mvn = self(X[n : n + self.gpu_batch_size].to(self.device))
            mu.append(mvn.mean.cpu())
            sigma.append(mvn.stddev.cpu())
        return torch.cat(mu, 0), torch.cat(sigma, 0)

    def embed_batched_gpu(self, X):
        # emb = torch.zeros((X.shape[0], self.architecture[-1]))
        # for n in range(0, X.shape[0], self.gpu_batch_size):
        #     emb[n:n+self.gpu_batch_size, :] = self.embedding(X[n : n + self.gpu_batch_size].to(self.device)).to(self.device)
        
        emb = torch.zeros((X.shape[0], self.architecture[-1]))
        for n in range(0, X.shape[0], self.gpu_batch_size):
            emb[n:n+self.gpu_batch_size, :] = self.embedding(X[n : n + self.gpu_batch_size].to(self.device)).detach()
        # print(emb[0].shape)
        return emb

    def eval_acquisition_batched_gpu(self, X, f=(lambda x: x)):
        acq = []
        for n in range(0, X.shape[0], self.gpu_batch_size):
            #acq.append(f(X[n : n + self.gpu_batch_size].to(self.device)).to(self.device))
            acq.append(f(X[n : n + self.gpu_batch_size].to(self.device)).detach())

        # print(emb[0].shape)
        return torch.cat(acq, 0)
    
    # def predict_batched_gpu(self, X, batch_size=1000):
    #     mu, sigma = [], []
    #     for n in range(0, X.shape[0], batch_size):
    #         mvn = self.posterior(X[n: n + batch_size].to(self.device))
    #         mu.append(mvn.mean.squeeze(1))
    #         sigma.append(mvn.stddev)
    #     return torch.cat(mu), torch.cat(sigma)

    # def embed_batched_gpu(self, X, batch_size=1000, f=(lambda x: x)):
    #     emb = []
    #     for n in range(0, X.shape[0], batch_size):
    #         emb.append(f(self.embedding(X[n: n + batch_size].to(self.device)).to(self.device)))
    #     return torch.cat(emb)

    def get_kernel_noise(self):
        return self.likelihood.noise.cpu().detach()