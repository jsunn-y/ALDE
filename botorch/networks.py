from __future__ import annotations

from collections.abc import Sequence, Mapping
from datetime import datetime
import time, os, sys
from typing import Literal
from dataclasses import dataclass, astuple
from utils import MapClass

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

# import nts

#comment out if not using BNNs
# import jax
# from jax import vmap
# import jax.numpy as jnp
# import jax.random as jrandom

# import numpyro
# import numpyro.distributions as npdist
# from numpyro.infer import (
#     MCMC as npMCMC,
#     NUTS as npNUTS,
#     init_to_feasible,
#     init_to_median,
#     init_to_sample,
#     init_to_uniform,
#     init_to_value,
# )

import pyro
import pyro.distributions as pdist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer.mcmc import NUTS as pNUTS, MCMC as pMCMC  # , HMC


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


# TODO: build this out, other types too
def GP_kernel_numpyro(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    # nothing happens w lengthscale here bc linear
    # just added in var scalar. before did not have.
    k = jnp.matmul(X, Z.T) # TODO: scale param, see gpytorch
    if include_noise:
        k += jnp.eye(X.shape[0]) * (jitter + noise)
    return var*k


# the non-linearity we use in our neural network
def nonlin_numpyro(x):
    return jax.nn.relu(x)


def model_fn(X, Y, architecture, kernel):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", npdist.LogNormal(0.0, 10.0))
    # TODO: would like a constraint on this.
    noise = numpyro.sample("kernel_noise", npdist.LogNormal(1e-6, 10.0))
    length = numpyro.sample("kernel_length", npdist.LogNormal(0.0, 10.0))

    z, arc = X, architecture
    for i in range(len(arc) - 1):
        w = numpyro.sample(
            f"w{i+1}",
            npdist.Normal(
                jnp.zeros((arc[i], arc[i + 1])), jnp.ones((arc[i], arc[i + 1]))
            ),
        )
        b = numpyro.sample(
            f"b{i+1}", npdist.Normal(jnp.zeros((arc[i + 1])), jnp.ones((arc[i + 1])))
        )
        z = nonlin_numpyro(jnp.matmul(z, w) + b)

    # compute kernel
    k = GP_kernel_numpyro(z, z, var, length, noise)

    # TODO: do we need another noise, or ok b/c kernel noise?
    # noise = numpyro.sample("noise", npdist.Uniform(0, 1))

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        npdist.MultivariateNormal(loc=jnp.zeros(z.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# do GP prediction for a given set of hyperparameters.
# uses well-known formula for gaussian process prediction
def predict_fn(rng_key, X, Y, X_test, params_sample):
    var, length, noise = (
        params_sample["kernel_var"],
        params_sample["kernel_length"],
        params_sample["kernel_noise"],
    )

    assert (len(params_sample.keys()) - 3) % 2 == 0
    num_layers = (len(params_sample.keys()) - 3) // 2
    for i in range(num_layers):
        w, b = params_sample[f"w{i+1}"], params_sample[f"b{i+1}"]
        X = nonlin_numpyro(jnp.matmul(X, w) + b)
        X_test = nonlin_numpyro(jnp.matmul(X_test, w) + b)

    # compute kernels between train and test data, etc.
    k_XX = GP_kernel_numpyro(X, X, var, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)

    k_pp = GP_kernel_numpyro(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = GP_kernel_numpyro(X_test, X, var, length, noise, include_noise=False)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jrandom.normal(
        rng_key, X_test.shape[:1]
    )
    return mean, sigma_noise


def get_samp_cov(rng_key, X, params_sample):
    var, length, noise = (
        params_sample["kernel_var"],
        params_sample["kernel_length"],
        params_sample["kernel_noise"],
    )

    assert (len(params_sample.keys()) - 3) % 2 == 0
    num_layers = (len(params_sample.keys()) - 3) // 2
    for i in range(num_layers):
        w, b = params_sample[f"w{i+1}"], params_sample[f"b{i+1}"]
        X = nonlin_numpyro(jnp.matmul(X, w) + b)

    # compute kernels between train and test data, etc.
    k_XX = GP_kernel_numpyro(X, X, var, length, noise, include_noise=True)
    k_XX_inv = jnp.linalg.inv(k_XX)

    return k_XX, k_XX_inv, X


# X passed in is embedding
def quick_predict_fn(rng_key, X, Y, X_test, params_sample, k_XX, k_XX_inv):
    var, length, noise = (
        params_sample["kernel_var"],
        params_sample["kernel_length"],
        params_sample["kernel_noise"],
    )

    assert (len(params_sample.keys()) - 3) % 2 == 0
    num_layers = (len(params_sample.keys()) - 3) // 2
    for i in range(num_layers):
        w, b = params_sample[f"w{i+1}"], params_sample[f"b{i+1}"]
        X_test = nonlin_numpyro(jnp.matmul(X_test, w) + b)

    # compute kernels between train and test data, etc.
    k_pp = GP_kernel_numpyro(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = GP_kernel_numpyro(X_test, X, var, length, noise, include_noise=False)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(k_XX_inv, k_pX.T))
    mean = jnp.matmul(k_pX, jnp.matmul(k_XX_inv, Y))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jrandom.normal(
        rng_key, X_test.shape[:1]
    )
    return mean, sigma_noise


# TODO: c/p another w/ just BNN, can we have it inherit so use same train, predict fns?
# TODO: also add in support for arb number of layers (see online tut/file)
class BDKL_MCMC_numpyro:  # inherit?
    def __init__(self, architecture, activation, inference_args, *_, **__):
        # TODO: make model fn do some params and then return the actual model fn?
        # kernel fn and nonlin too
        self.model_fn = lambda X, Y: model_fn(X, Y, architecture, None)
        self.dkl, self.bnn = True, True
        self.ind = None
        self.inference_args = inference_args
        self.rng_key = jrandom.PRNGKey(np.random.randint(0, sys.maxsize))

    def train_model(
        self,
        X,
        Y,
        num_warmup: int = 100,
        num_samples: int = 100,
        num_chains: int = 1,
        thinning: int = 2,
        verbose=2,
        *_,
        **__,
    ):
        # TODO: put this in fn call above
        self.X, self.Y = X.cpu().numpy(), Y.cpu().numpy()

        self.rng_key, rng_key_run = jrandom.split(self.rng_key)
        start = time.time()
        init_strategy = init_to_median(num_samples=10)
        kernel = npNUTS(self.model_fn, init_strategy=init_strategy)
        mcmc = npMCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            thinning=thinning,
            progress_bar=(verbose>=3),
            # chain_method="vectorized",
        )
        # TODO: load in old samples if exist?

        mcmc.run(rng_key_run, self.X, self.Y)
        # mcmc.print_summary()
        if verbose >= 2: print("\nMCMC elapsed time:", time.time() - start)
        self.samples = mcmc.get_samples()

        return self, None  # likelihood

    def predict(self, X_test, Y_test=None):
        # return self.predict_averaging(X_test, Y_test=Y_test)
        return self.predict_sample(X_test, Y_test=Y_test)

    def predict_averaging(self, X_test, Y_test=None):
        self.rng_key, rng_key_predict = jrandom.split(self.rng_key)
        samples = self.samples

        # do prediction
        vmap_args = (
            jrandom.split(rng_key_predict, samples["kernel_var"].shape[0]),
            samples,
        )
        means, sigma = vmap(
            lambda rkey, sample: predict_fn(
                rkey,
                self.X,
                self.Y,
                X_test.cpu().numpy(),
                sample,
            )
        )(*vmap_args)

        mean_prediction = np.mean(means, axis=0)
        std_prediction = np.mean(sigma, axis=0)
        # pred = means + sigma_noise
        # percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
        if Y_test != None:
            test_mse = np.mean((mean_prediction - Y_test.cpu().numpy()) ** 2)  # MSE
            # print(f'BNN Test MSE: {test_mse}')
            return (
                torch.tensor(np.array(mean_prediction)),
                torch.tensor(np.array(std_prediction)),
                test_mse,
            )
        return (
            torch.tensor(np.array(mean_prediction)),
            torch.tensor(np.array(std_prediction)),
            None,
        )

    def get_num_samples(self):
        return self.samples["kernel_var"].shape[0]

    def draw_random_seed(self):
        self.ind = np.random.randint(0, high=self.get_num_samples())

    def sample_params(self, index=None):
        if self.ind is None:
            self.draw_random_seed()
        return self.get_sample(self.ind)

    def get_sample(self, index):
        single_sample = {}
        for key in self.samples:
            single_sample[key] = self.samples[key][index]
        return single_sample

    def predict_sample(self, X_test, Y_test=None):
        self.rng_key, rng_key_predict = jrandom.split(self.rng_key)

        # do prediction based on seed drawn above
        single_sample = self.sample_params()
        mean, sigma = predict_fn(
            rng_key_predict, self.X, self.Y, X_test.cpu().numpy(), single_sample
        )

        if Y_test != None:
            test_mse = np.mean((mean - Y_test.cpu().numpy()) ** 2)  # MSE
            # print(f'BNN Test MSE: {test_mse}')
            return torch.tensor(np.array(mean)), torch.tensor(np.array(sigma)), test_mse
        return torch.tensor(np.array(mean)), torch.tensor(np.array(sigma)), None

    def predict_sample_batch(self, X_test, Y_test=None, batch_size=1000):
        self.rng_key, rng_key_predict = jrandom.split(self.rng_key)
        # do prediction based on seed drawn above
        single_sample = self.sample_params()
        k_XX, k_XX_inv, emb = get_samp_cov(rng_key_predict, self.X, single_sample)
        X_test = X_test.cpu().numpy()
        mu, sigma = [], []
        for n in range(0, X_test.shape[0], batch_size):
            mean, sd = quick_predict_fn(
                rng_key_predict,
                emb,
                self.Y,
                X_test[n : n + batch_size],
                single_sample,
                k_XX,
                k_XX_inv,
            )
            mu.append(torch.tensor(np.array(mean)))
            sigma.append(torch.tensor(np.array(sd)))
        return torch.cat(mu, 0), torch.cat(sigma, 0), None

        # if Y_test != None:
        #     test_mse = np.mean((mean-Y_test.cpu().numpy())**2) # MSE
        #     # print(f'BNN Test MSE: {test_mse}')
        #     return torch.tensor(np.array(mean)), torch.tensor(np.array(sigma)), test_mse
        # return torch.tensor(np.array(mean)), torch.tensor(np.array(sigma)), None

    # this is based on a single random sample of params
    def embedding(self, X_test):
        single_sample = self.sample_params()

        assert (len(single_sample.keys()) - 3) % 2 == 0
        num_layers = (len(single_sample.keys()) - 3) // 2
        emb = X_test.cpu().numpy()
        for i in range(num_layers):
            w, b = single_sample[f"w{i+1}"], single_sample[f"b{i+1}"]
            emb = nonlin_numpyro(jnp.matmul(emb, w) + b)
        return torch.tensor(np.asarray(emb))

    # usually calling in conjunction w embedding or preds, so want same sample
    def get_kernel_noise(self):
        # TODO: want noise constraint. also, bring this in from min_noise
        noise_lim = 1e-4
        single_sample = self.sample_params()
        if single_sample["kernel_noise"] > noise_lim:
            return torch.tensor(np.asarray(single_sample["kernel_noise"]))
        else:
            return torch.tensor(np.asarray(jnp.array([noise_lim])[0]))


######################################


class BNN_base_pyro(PyroModule):
    """Uses PyroModule[nn.Linear] instead of our own BayesianLinear class."""

    def __init__(self, dims, device="cuda"):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layer = PyroModule[nn.Linear](dims[i], dims[i + 1])
            layer.weight = PyroSample(
                prior=pdist.Normal(*torch.tensor([0.0, 1.0]).to(device))
                .expand([dims[i + 1], dims[i]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                prior=pdist.Normal(*torch.tensor([0.0, 1.0]).to(device))
                .expand([dims[i + 1]])
                .to_event(1)
            )
            layers.append(layer)
        self.layers = PyroModule[torch.nn.ModuleList](layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.relu(x)
        return x


class BNN_pyro(PyroModule):
    def __init__(self, dims, device="cuda"):
        super().__init__()
        assert dims[-1] == 1
        self.bnn = BNN_base_pyro(dims, device)

    def forward(self, x, y=None):
        mu = self.bnn(x).squeeze()
        sigma = pyro.sample("sigma", pdist.Uniform(0, 1))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", pdist.Normal(mu, sigma), obs=y)
        return mu


class GP_pyro(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood: gpytorch.likelihoods.Likelihood,
        dims: Sequence[int],
        device="cuda",
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = BNN_base_pyro(dims, device)

        # This module will scale the NN features so that they're nice values
        # TODO: do we need to, want to do this?
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: Tensor) -> gdist.MultivariateNormal:
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        # TODO: check if this makes diff, should we be doing with standard DKL?
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gdist.MultivariateNormal(mean_x, covar_x)


class BDKL_SVI_pyro:
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        architecture: Sequence[int],
        device: torch.device | str = "cuda",
        *_,
        **__,
    ):
        self.likelihood = likelihood
        self.architecture = architecture
        self.dkl, self.bnn = True, True
        self.device = device

        self.train_x = train_x.to(device, dtype=torch.float)
        self.train_y = train_y.to(device, dtype=torch.float)

        # TODO: this probably has to get remade when training data changes
        self.gpmodel = GP_pyro(
            train_x=self.train_x,
            train_y=self.train_y,
            likelihood=likelihood,
            dims=architecture,
            device=device,
        ).to(device)

        self.gpmodel.mean_module.register_prior(
            name="mean_prior",
            prior=UniformPrior(*torch.tensor([-1.0, 1.0]).to(device)),
            param_or_closure="constant",
        )
        self.gpmodel.covar_module.base_kernel.register_prior(
            name="lengthscale_prior",
            prior=UniformPrior(*torch.tensor([0.01, 0.5]).to(device)),
            param_or_closure="lengthscale",
        )
        self.gpmodel.covar_module.register_prior(
            name="outputscale_prior",
            prior=UniformPrior(*torch.tensor([1.0, 2.0]).to(device)),
            param_or_closure="outputscale",
        )
        self.likelihood.register_prior(
            name="noise_prior",
            prior=UniformPrior(*torch.tensor([0.01, 0.5]).to(device)),
            param_or_closure="noise",
        )

        # define a pyro model, which can be passed into SVI, MCMC functions
        # TODO: have a function that takes current gpmodel and then spits out this function
        def pyro_model(x, y=None):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.gpmodel.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("obs", output, obs=y)
            return y

        self.pyro_model = pyro_model

    def train_model(self, X: Tensor, Y: Tensor, lr: float, num_iter: int=20_000, *_, **__):
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.pyro_model)

        svi = pyro.infer.SVI(
            self.pyro_model,
            self.guide,
            pyro.optim.Adam({"lr": lr}),  # 1e-2
            pyro.infer.Trace_ELBO(),
        )

        X = X.to(self.device, dtype=torch.float)
        Y = Y.to(self.device, dtype=torch.float)

        pyro.clear_param_store()
        self.gpmodel.train()
        self.gpmodel.set_train_data(X, Y, strict=False)
        for i in range(num_iter):
            elbo = svi.step(X, Y)
            # TODO: add a verbosity arg here
            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}, Elbo loss: {elbo}")
            # TODO: clear gpu cache?
        return self, None

    def predict(
        self, X_test: Tensor, Y_test: Tensor | None = None, num_samples: int = 50
    ) -> tuple[Tensor, Tensor]:
        """
        Args
            X_test: shape [n, d], inputs
            Y_test: shape [n], optional labels (for printing MSE)
            num_samples: number of samples to draw

        Returns
            mean_preds: shape [n], average mean function at each point
            std_preds: shape [n], average std-dev at each point
        """
        self.gpmodel.eval()
        X_test = X_test.to(self.device, dtype=torch.float)

        # each time gpmodel forward() gets run, draws a bunch of samples
        means = []
        stds = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.gpmodel(
                    X_test
                )  # this is a gpytorch.distributions.MultivariateNormal object
            means.append(output.mean.detach())
            stds.append(output.stddev.detach())

        mean_preds = torch.stack(means, dim=-1)  # shape [n, num_samples]
        mean_preds = torch.mean(mean_preds, dim=-1)  # shape [n]
        std_preds = torch.stack(stds, dim=-1)  # shape [n, num_samples]
        std_preds = torch.mean(std_preds, dim=-1)  # shape [n]

        if Y_test is not None:
            Y_test = Y_test.to(self.device)
            test_mse = torch.mean((mean_preds - Y_test) ** 2)
            print(f"BNN Test MSE: {test_mse.item()}")
            return mean_preds, std_preds, test_mse

        return mean_preds, std_preds, None


# Not updated
class BDKL_MCMC_pyro:
    def __init__(self, train_x, train_y, likelihood, architecture):
        self.likelihood, self.architecture = likelihood, architecture
        self.train_x, self.train_y = train_x, train_y
        # TODO: this probably has to get remade when training data changes
        self.gpmodel = GP_pyro(train_x, train_y, likelihood, architecture)
        self.gpmodel.mean_module.register_prior(
            "mean_prior", UniformPrior(-1, 1), "constant"
        )
        self.gpmodel.covar_module.base_kernel.register_prior(
            "lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale"
        )
        self.gpmodel.covar_module.register_prior(
            "outputscale_prior", UniformPrior(1, 2), "outputscale"
        )
        self.likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")
        self.dkl, self.bnn = True, True

        # define a pyro model, which can be passed into SVI, MCMC functions
        # TODO: have a function that takes current gpmodel and then spits out this function
        def pyro_model(x, y=None):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.gpmodel.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                # pyro.sample("obs", output, obs=y)
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("obs", output, obs=y)
            return y

        self.pyro_model = pyro_model

    def train_model(self, X, Y, lr, num_iter, aux=None):
        # TODO: have aux == MCMC params like num samples, warmup steps, etc.
        num_samples, warmup_steps = 5, 5

        kernel = pNUTS(self.pyro_model)
        mcmc_run = pMCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        # TODO: add verbosity param here too?
        mcmc_run.run(X, Y)

        self.samples = mcmc_run.get_samples()

    def predict(self, X_test, Y_test=None, num_samples=50):
        # here, num_samples is likely desired to be however many we drew during inference
        # TODO: add option for num_samples to just be 'MAX'
        self.gpmodel.eval()
        expanded_test_x = X_test.unsqueeze(0).repeat(num_samples, 1, 1)
        preds = self.gpmodel(expanded_test_x)
        mean_preds = preds.mean.detach()
        std_preds = preds.stddev.detach()
        print(mean_preds.shape, std_preds.shape)

        if Y_test != None:
            test_mse = torch.mean((mean_preds - Y_test) ** 2)
            print(f"BNN Test MSE: {test_mse.detach()}")

        return mean_preds, std_preds


######################################

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
        self.dkl, self.bnn = True, False

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
        self.dkl, self.bnn = True, False
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
        self.dkl, self.cdkl, self.bnn = False, False, False
        self.device = device
        self.architecture = architecture

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
            #fix this later so it only does this for the CNN
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
        if self.feature_extractor != None:
            self.feature_extractor.train()
        #adam is a first order optimization, LBFGSB is a better optimization but requires a hessian
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
            for iter in range(num_iter):
                optimizer.zero_grad()
                # TODO: shouldn't be forward
                preds = self(X)
                loss = -mll(preds, Y)
                loss.backward()
                #print(loss)
                optimizer.step()

        if self.feature_extractor != None:
            self.feature_extractor.eval()
        self.eval()
        self.likelihood.eval()
        return None

    def get_kernel_noise(self):
        return self.likelihood.noise.cpu()

    def predict_batched_gpu(self, X, batch_size=1000):
        mu, sigma = [], []
        for n in range(0, X.shape[0], batch_size):
            # TODO: forward gives prior, model uses posterior
            mvn = self(X[n : n + batch_size].to(self.device))
            mu.append(mvn.mean.cpu())
            sigma.append(mvn.stddev.cpu())
        return torch.cat(mu, 0), torch.cat(sigma, 0)

    def embed_batched_gpu(self, X, batch_size=1000):
        emb = torch.zeros((X.shape[0], self.architecture[-1]))
        for n in range(0, X.shape[0], batch_size):
            emb[n:n+batch_size, :] = self.embedding(X[n : n + batch_size].to(self.device)).to(self.device)
        # print(emb[0].shape)
        return emb

    def eval_acquisition_batched_gpu(self, X, batch_size=1000, f=(lambda x: x)):
        acq = []
        for n in range(0, X.shape[0], batch_size):
            acq.append(f(X[n : n + batch_size].to(self.device)).to(self.device))
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
        
        self.device = device
        self.dkl, self.bnn, self.lin = False, False, False # lin False for now bc always matern
        # if kernel.lower() not in ['mat', 'matern']: raise NotImplementedError()
        self.feature_extractor = None
        if (len(architecture) > 2): 
            self.dkl = True

        if kernel == None or kernel.lower() == "rbf":
            covar_module = gpytorch.kernels.ScaleKernel(
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
            covar_module = gpytorch.kernels.LinearKernel()
            self.lin = True
        else: raise NotImplementedError(f"Add your kernel {kernel} in networks.py to use it.")

        SingleTaskGP.__init__(
            self,
            # TODO this doesn't work still bc training inputs diff
            train_X=train_x.float(),# if not self.dkl else torch.zeros(train_x.shape[0], architecture[-2]).float(),
            train_Y=train_y.unsqueeze(-1).float(),
            covar_module=covar_module,
            # TODO: unclear what should be done here for NN outputs, etc.
            # input_transform=input_transform if not dkl else None,
            outcome_transform=Standardize(m=1),
        )

        if self.dkl:
            # chop GPR part of arc off, from E --> 1
            self.feature_extractor = DNN_FF(architecture[:-1], activation, p_dropout)
        #need to add CNN here

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        # We're first putting our data through a deep net (feature extractor)
        emb = self.embedding(x)
        mean_x = self.mean_module(emb)
        covar_x = self.covar_module(emb)
        return gdist.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x: Tensor) -> Tensor:
        # for use with TS acq
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

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        if not self.dkl:
            self.likelihood, self = self.likelihood.cpu(), self.cpu()
            fit_gpytorch_mll(mll)
        else:
            self.feature_extractor.train()
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
                for iter in range(num_iter):
                    optimizer.zero_grad()
                    # IMPORTANT: don't use fwd
                    preds = self(X)
                    loss = -mll(preds, Y)
                    loss.backward()
                    optimizer.step()
            self.feature_extractor.eval()

        self.eval()
        self.likelihood.eval()
        return None

    def predict_batched_gpu(self, X, batch_size=1000):
        mu, sigma = [], []
        for n in range(0, X.shape[0], batch_size):
            mvn = self.posterior(X[n : n + batch_size].to(self.device))
            mu.append(mvn.mean.squeeze(1))
            sigma.append(mvn.stddev)
        return torch.cat(mu, 0), torch.cat(sigma, 0)

    def embed_batched_gpu(self, X, batch_size=1000, f=(lambda x: x)):
        emb = []
        for n in range(0, X.shape[0], batch_size):
            emb.append(f(self.embedding(X[n : n + batch_size].to(self.device)).to(self.device)))
        # print(emb[0].shape)
        return torch.cat(emb, 0)

    def get_kernel_noise(self):
        return self.likelihood.noise.cpu()
