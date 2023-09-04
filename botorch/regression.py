from __future__ import annotations

from typing import Literal

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
def get_data(obj, N=20, sigma_obs=0.15, N_test=400):
    np.random.seed(0)

    # obj_fn = obj.objective
    # domain = obj.get_domain()
    # ymax = obj.get_max()
    # disc_X = obj.get_points()[0]
    X, Y = utils.samp_discrete(N, obj)
    X_test, Y_test = utils.samp_discrete(N_test, obj)
    Y, Y_test = (Y - torch.mean(Y))/torch.std(Y), (Y_test - torch.mean(Y))/torch.std(Y)

    return X.cuda(), Y.cuda(), X_test.cuda(), Y_test.cuda()


def main(args):
    n_data_subsamples = 10
    ns, nw, nc = 100, 100, 1

    obj = objectives.FoldX

    mses = torch.zeros(len(range(5,305,5)), 5) # Amean, Astddev, Smean, Sstddev, time

    # BDKL -- MCMC numpyro #####################
    # numpyro.set_platform('gpu')
    for i, n_data in enumerate(range(5,305,5)[::-1]):#[::-1]):
        Anmse, Snmse = [], []
        times = []
        for _ in range(n_data_subsamples):
            X, Y, X_test, Y_test = get_data(obj, N=n_data)

            start = time.time()
            bdkl = networks.BDKL_MCMC_numpyro([X.shape[-1], 25], None, None, inp_dim=None)
            # conversion to numpy internally, don't need to convert back here
            bdkl.train_model(X, Y, nw, ns, nc, verbose=True)
            times.append(time.time()-start)
            _,_, Amse = bdkl.predict_averaging(X_test, Y_test=Y_test)
            _,_, Smse = bdkl.predict_sample(X_test, Y_test=Y_test)
            Anmse.append(Amse.item())
            Snmse.append(Smse.item())
        Anmse, Snmse = torch.tensor(Anmse), torch.tensor(Snmse)
        times = torch.tensor(times)
        mses[i][0], mses[i][1], mses[i][2], mses[i][3], mses[i][4] = torch.mean(Anmse), torch.std(Anmse), torch.mean(Snmse), torch.std(Snmse), torch.mean(times)
        print(f"\n----\n{n_data} pts: {mses[i]} MSE\n------\n")
        torch.save(mses, 'bdkl_reg_mses_numpyro_vec.pt')
    
    # print('---------')
    # print(mses)

    # BDKL -- SVI pyro #####################
    # for i, n_data in enumerate(range(5,305,5)):#[::-1]):
    #     Anmse, Snmse = [], []
    #     times = []
    #     for _ in range(n_data_subsamples):
    #         X, Y, X_test, Y_test = get_data(N=n_data)

    #         start = time.time()
    #         lhood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0)).cuda()
    #         bdkl = networks.BDKL_SVI_pyro(X, Y, lhood, [X.shape[-1], 25])
    #         # conversion to numpy internally, don't need to convert back here
    #         bdkl.train_model(X, Y, 1e-1, 1_000)
    #         times.append(time.time()-start)
    #         _,_, Amse = bdkl.predict(X_test, Y_test=Y_test, num_samples=50)
    #         _,_, Smse = bdkl.predict(X_test, Y_test=Y_test, num_samples=1)
    #         Anmse.append(Amse.cpu())
    #         Snmse.append(Smse.cpu())
    #     Anmse, Snmse = torch.tensor(Anmse), torch.tensor(Snmse)
    #     times = torch.tensor(times)
    #     mses[i][0], mses[i][1], mses[i][2], mses[i][3], mses[i][4] = torch.mean(Anmse), torch.std(Anmse), torch.mean(Snmse), torch.std(Snmse), torch.mean(times)
    #     print(f"\n----\n{n_data} pts: {mses[i]} MSE\n------\n")
    #     torch.save(mses, 'bdkl_reg_mses_pyro_1000iter_1e-1lr.pt')
    
    # print('---------')
    # print(mses)

    # DKL #####################
    # for i, n_data in enumerate(range(5,305,5)):#[::-1]:
    #     nmse = []
    #     times = []
    #     for _ in range(n_data_subsamples):
    #         X, Y, X_test, Y_test = get_data(obj, N=n_data)
    #         print(n_data, _)
    #         start = time.time()
    #         # arc = [X.shape[-1], 500, 150, 50, 1]
    #         arc = [X.shape[-1], 30, 1]
    #         lhood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).cuda()
    #         dkl = networks.GP(X, Y, lhood, 'lin', arc, 'relu').cuda()
    #         # TODO: to test if DNN params actually getting updated
    #         # print(list(dkl.feature_extractor.parameters()))
    #         dkl.train_model(X, Y, 1e-1, 100)
    #         print('\n\n')
    #         # print(list(dkl.feature_extractor.parameters()))
    #         dkl.eval()
    #         lhood.eval()
    #         times.append(time.time()-start)

    #         # preds = dkl(X_test)
    #         pred_mean, _ = dkl.predict_batched_gpu(X_test, batch_size=10)
    #         # mse = torch.mean((preds.mean.cpu()-Y_test.cpu())**2)
    #         mse = torch.mean((pred_mean-Y_test.cpu())**2)
    #         nmse.append(mse)
    #     nmse = torch.tensor(nmse)
    #     times = torch.tensor(times)
    #     mses[i][0], mses[i][1], mses[i][2] = torch.mean(nmse), torch.std(nmse), torch.mean(times)
    #     print(f"{n_data} pts: {mses[i]} MSE")
    #     torch.save(mses, 'dkl_reg_gb1_30d.pt')
    
    print('---------')
    print(mses)




# TODO: change this stuff
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

    # assert numpyro.__version__.startswith("0.10.1")
    print(numpyro.__version__)
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