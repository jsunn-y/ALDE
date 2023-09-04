import argparse
import os
import time
from datetime import datetime

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


def main(args):

    print(f'args seed {args.seed}')

    if args.obj.lower() == 'foldx':
        obj = objectives.FoldX
    elif args.obj.lower() == 'gb1':
        obj = objectives.GB1
    elif args.obj.lower() == 'nano':
        obj = objectives.Nanophotonics
    else:
        raise NotImplementedError()
        # obj = objectives.Hartmann_3d

    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]

    randx, randy = utils.samp_discrete(args.num_data, obj)

    mtype = 'BDKL'
    kernel = args.kernel
    acq = args.acq
    arc = (domain[0].size(-1), 30,)
    fname = mtype + 'sweep-' + str(10) + 'DO-' + str(0) + '-' + kernel + '-' + acq + '_' + str(args.index + 1)
    print(fname)
    BayesianOptimization.run(obj_fn, domain, acq, arc, 'relu', 0, mtype, 1, 0, 0, 100, 100, 0.1, .01, .01, kernel, None, args.budget, .01, None, None, randx, randy, None, None, None, disc_X, ymax, False, args.path + fname, 1, args.seed, 1)
    
    print(f'Done: {fname}')    


if __name__ == "__main__":
    
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # assert numpyro.__version__.startswith("0.10.1")
    parser = argparse.ArgumentParser(description="Bayesian DKBO launcher")

    # actual exp params
    parser.add_argument("-i", "--index", nargs="?", default=-1, type=int)
    parser.add_argument("-b", "--budget", nargs="?", default=300, type=int)
    parser.add_argument("-k", "--kernel", nargs="?", default='Lin', type=str)
    parser.add_argument("-a", "--acq", nargs="?", default='UCB', type=str)
    parser.add_argument("-o", "--obj", nargs="?", default='foldx', type=str)
    parser.add_argument("-p", "--path", nargs="?", default='./test/results/paper_2022/botorch/bnn_test_fx/', type=str)


    # parser.add_argument("-n", "--num-samples", nargs="?", default=100, type=int)
    # parser.add_argument("--num-warmup", nargs="?", default=100, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    # parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=10, type=int) # init points, rand sampled
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    # parser.add_argument(
    #     "--init-strategy",
    #     default="median",
    #     type=str,
    #     choices=["median", "feasible", "value", "uniform", "sample"],
    # )
    # parser.add_argument("--pname", default="gp_plot.png", type=str)
    args = parser.parse_args()

    # make dirs
    os.makedirs(args.path, exist_ok=True)

    # Seeding stuff
    with open('../rndseed.txt', 'r') as f:
        lines = f.readlines()
        print('run index: {}'.format(args.index))
        line = lines[args.index].strip('\n')
        print('seed: {}'.format(int(line)))
        seed = int(line)
        args.seed = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # IMPORTANT!!! use CPU for MCMC
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)