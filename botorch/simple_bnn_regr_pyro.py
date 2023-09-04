import argparse
import os
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

import objectives, utils, networks

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

import gpytorch
import torch

matplotlib.use("Agg")  # noqa: E402


# pull some data from real dataset (can also have this be artificial)
def get_data(N=20, sigma_obs=0.15, N_test=400):
    np.random.seed(0)

    obj = objectives.FoldX
    # obj_fn = obj.objective
    # domain = obj.get_domain()
    # ymax = obj.get_max()
    # disc_X = obj.get_points()[0]
    X, Y = utils.samp_discrete(N, obj)
    X_test, Y_test = utils.samp_discrete(N_test, obj)
    Y, Y_test = (Y - torch.mean(Y))/torch.std(Y), (Y_test - torch.mean(Y))/torch.std(Y)

    return X.numpy(), Y.numpy(), X_test.numpy(), Y_test.numpy()


def main(args):
    # for i in range(args.num_data, args.num_data+10):
    i = args.num_data
    X, Y, X_test, Y_test = get_data(N=i)
    print(f'Data: X {X.shape}, X_test {X_test.shape}, Y {Y.shape}, Y_test {Y_test.shape}')

    # normal DKL training + MSE (convert to torch, cuda here)
    X, Y, X_test, Y_test = torch.tensor(X).cuda(), torch.tensor(Y).cuda(), torch.tensor(X_test).cuda(), torch.tensor(Y_test).cuda()
    arc = [X.shape[-1], 500, 50, 5]
    lhood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0)).cuda()
    dnn = networks.DNN_FF(arc, 'relu', 0, inp_dim=arc[0]).cuda()
    dkl = networks.GP(X, Y, lhood, arc[-1], 'lin', None, None, dkl=dnn, bnn=False).cuda()
    dkl.train_model(X, Y, .01, 100, aux=lhood)
    dkl.eval()
    lhood.eval()

    preds = dkl(X_test)
    nn_mse = torch.mean((preds.mean.cpu()-Y_test.cpu())**2)
    print(f'NN DKL Test MSE: {nn_mse}')



    # **BNN*** do inference
    # none of the params actually matter currently
    # bdkl = networks.BNN_MCMC(None, None, None, inp_dim=None)
    # # conversion to numpy internally, don't need to convert back here
    # bdkl.train_model(X, Y, None, None, None)
    # samples = bdkl.predict(X_test, Y_test=Y_test)

    # PYRO inference

    # you should do more than 100 training iter
    new_lhood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0))
    new_arc = [X.shape[-1], 25, 5]
    bdkl = networks.BDKL_SVI_pyro(X, Y, new_lhood, new_arc)
    bdkl.train_model(X, Y, 1e-2, 100, aux=None)
    samples = bdkl.predict(X_test, Y_test=Y_test)

    # TODO: switch cuda devices
    new_lhood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0))
    new_arc = [X.shape[-1], 25, 5]
    bdkl = networks.BDKL_MCMC_pyro(X.cpu(), Y.cpu(), new_lhood, new_arc)
    bdkl.train_model(X.cpu(), Y.cpu(), None, None, aux=None)
    samples = bdkl.predict(X_test.cpu(), Y_test=Y_test.cpu())

    # # make plots
    # fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # # plot training data
    # ax.plot(X[:,0], Y, "kx")
    # # plot 90% confidence level of predictions
    # ax.fill_between(X_test[:,0], percentiles[0, :], percentiles[1, :], color="lightblue")
    # # plot mean prediction
    # ax.plot(X_test[:,0], mean_prediction, "blue", ls="solid", lw=2.0)
    # ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    # plt.savefig(args.pname)


if __name__ == "__main__":
    seed = 402
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    assert numpyro.__version__.startswith("0.10.1")
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=100, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=100, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    parser.add_argument("--pname", default="gp_plot.png", type=str)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)