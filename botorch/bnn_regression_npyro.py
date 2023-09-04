import argparse
import os
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

import objectives, utils, networks, models
from optimize import BayesianOptimization

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

def is_pos_sdef(x):
    # print(np.linalg.eigvals(x.detach()))
    return jnp.all(jnp.linalg.eigvals(x) >= 0)

def is_pos_def(A: np.ndarray) -> bool:
    """Checks whether a matrix is positive definite.
    Args
    - A: np.array, matrix
    Returns: bool, true iff A>0
    """
    if np.array_equal(A, A.T):
        try:
            # np.linalg.cholesky(A.detach())
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

# the non-linearity we use in our neural network
def nonlin(x):
    return jax.nn.relu(x)
    # return torch.nn.ReLU(x)
    # return jnp.tanh(x) # swap for relu?

# squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    # print('kernel')
    # X, Z = X.reshape((-1, 1)), Z.reshape((-1, 1))
    # print(X.shape, Z.shape)
    # deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    # deltaXsq = jnp.linalg.norm((X - Z), axis=)
    # k = var * jnp.exp(-0.5 * deltaXsq)
    # if include_noise:
    #     k += (noise + jitter) * jnp.eye(X.shape[0])
    # kern = gpytorch.kernels.RBFKernel()
    # print(kern(torch.tensor(X), torch.tensor(Z)).shape)
    # print(jnp.matmul(X, Z.T).shape)
    # print(jnp.einsum('ab,cb->abc',X,Z).shape)
    # print(jnp.eye(X.shape[0]).shape)
    k = jnp.matmul(X, Z.T)
    if include_noise:
        k += jnp.eye(X.shape[0]) * (jitter + noise)
    
    # covar_module = gpytorch.kernels.LinearKernel()
    # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # print(X)
    # print(type(X))
    # X.ndimension = lambda : len(X.shape)
    # print(X.ndimension)
    # Xtorch = torch.tensor(X)
    # print(Xtorch, type(Xtorch))
    # k = covar_module(X).to_dense()
     # .to_dense()

    # print(f'k {k.shape}') #; PSD? {is_pos_sdef(k)}')
    # # print(f'PSD? {is_pos_sdef(k)}')
    # print(f'PD? {is_pos_def(k)}')
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    # noise = 1
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # # try BNN stuff
    # D_H, D_Y = 10, 5#1
    D_H1, D_H2, D_H3 = 25, 100, 10
    N, D_X = X.shape

    # sample 1st layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H1)), jnp.ones((D_X, D_H1))))
    assert w1.shape == (D_X, D_H1)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((D_H1)), jnp.ones((D_H1))))
    z1 = nonlin(jnp.matmul(X, w1) + b1)  # <= 1st layer of activations
    assert z1.shape == (N, D_H1)
    # # sample 2nd layer (we put unit normal priors on all weights)
    # w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H1, D_H2)), jnp.ones((D_H1, D_H2))))
    # assert w2.shape == (D_H1, D_H2)
    # b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((D_H2)), jnp.ones((D_H2))))
    # z2 = nonlin(jnp.matmul(z1, w2) + b2)  # <= 1st layer of activations
    # assert z2.shape == (N, D_H2)
    # # sample 3rd layer (we put unit normal priors on all weights)
    # w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H2, D_H3)), jnp.ones((D_H2, D_H3))))
    # assert w3.shape == (D_H2, D_H3)
    # b3 = numpyro.sample("b3", dist.Normal(jnp.zeros((D_H3)), jnp.ones((D_H3))))
    # z3 = nonlin(jnp.matmul(z2, w3) + b3)  # <= 3rd layer of activations
    # assert z3.shape == (N, D_H3)
    z3 = z1

    # ###########

    # compute kernel
    # k = kernel(X, X, var, length, noise)
    k = kernel(z3, z3, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(z3.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"kernel_var": 1.0, "kernel_noise": 0.05, "kernel_length": 0.5}
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    # mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, noise, w1, b1):#, w2, b2, w3, b3):
    # TODO: port BNN as feature extractor here, preprocess X
    X = nonlin(jnp.matmul(X, w1) + b1)
    X_test = nonlin(jnp.matmul(X_test, w1) + b1)
    # X = nonlin(jnp.matmul(X, w2) + b2)
    # X_test = nonlin(jnp.matmul(X_test, w2) + b2)
    # X = nonlin(jnp.matmul(X, w3) + b3)
    # X_test = nonlin(jnp.matmul(X_test, w3) + b3)
    # print(f'post emb shapes {X.shape, X_test.shape}')

    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    # print('after kernel calcs')
    # make torch
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jrandom.normal(
        rng_key, X_test.shape[:1]
    )
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


# create artificial regression dataset
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
    X, Y, X_test, Y_test = get_data(N=args.num_data)
    print(f'Data: X {X.shape}, X_test {X_test.shape}, Y {Y.shape}, Y_test {Y_test.shape}')

    # TODO: normal DKL training + MSE
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
    # bdkl = networks.BNN_MCMC(None, None, None, inp_dim=None)
    # bdkl.train_model(X, Y, None, None, None)

    # m = models.Model(X, Y, None, 100, None, 'BDKL', 'Lin', None, 'relu', 0, 0, .01, True)
    # bm, _,_,_ = m.train(X, Y)
    # bdkl = bm.model
    # samples = bdkl.predict(X_test, Y_test=Y_test)

    #######################

    # obj = objectives.FoldX
    # # obj = objectives.Hartmann_3d
    # obj_fn = obj.objective
    # domain = obj.get_domain()
    # ymax = obj.get_max()
    # disc_X = obj.get_points()[0]

    # randx, randy = utils.samp_discrete(10, obj)

    # mtype = 'BDKL'
    # kernel = 'Lin'
    # acq = 'UCB'
    # fname = mtype + 'sweep-' + str(0) + 'DO-' + str(0) + '-' + kernel + '-' + acq + '_' + str(0 + 1)
    # print(fname)
    # BayesianOptimization.run(obj_fn, domain, acq, None, 'relu', 0, mtype, 1, 0, 0, 100, 100, 0.1, .01, .01, kernel, None, 100, .01, None, None, randx, randy, None, None, None, disc_X, ymax, True, 'test123', 1, 1234, 1)
    
    ########################
    X, Y, X_test, Y_test = X.cpu().numpy(), Y.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy()
    rng_key, rng_key_predict = jrandom.split(jrandom.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y)
    print('post inference')

    # do prediction
    vmap_args = (
        jrandom.split(rng_key_predict, samples["kernel_var"].shape[0]),
        samples["kernel_var"],
        samples["kernel_length"],
        samples["kernel_noise"],
        samples["w1"],        
        samples["b1"],
        # samples["w2"],        
        # samples["b2"],
        # samples["w3"],        
        # samples["b3"],        
    )
    means, predictions = vmap(
        lambda rng_key, var, length, noise, w1, b1, #w2, b2, w3, b3
        : predict(rng_key, X, Y, X_test, var, length, noise, 
        w1, b1, #w2, b2, w3, b3,
        )
    )(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
    print(mean_prediction.shape, predictions.shape, means.shape)
    # TODO: MSE -- is predictions over test_x?
    test_mse = np.mean((mean_prediction-Y_test)**2)
    print(f'BNN Test MSE: {test_mse}')


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