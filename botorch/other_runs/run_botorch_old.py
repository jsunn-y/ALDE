from __future__ import annotations
import torch
import gpytorch
import botorch

import numpy as np
import random
from datetime import datetime
import glob
import os
import math
import multiprocessing as mp
import warnings

from optimize import BayesianOptimization
import objectives
import utils

from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
# from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

'''
Sample experiment runner script for DK-BO. Launches optimization runs as
separate processes.
This uses the Hartmann 6D dataset as an example.
'''

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # USER: set up objective func and related values
    #   -- add your own objective function in objectives.py
    #   -- OR replace obj_fn with function handle, etc...
    # fun = botorch.test_functions.synthetic.Hartmann(dim=6, negate=True)#.to(dtype=dtype, device=device)
    # fun.bounds[0, :].fill_(0)
    # fun.bounds[1, :].fill_(1)
    # # dim = fun.dim

    # obj_fn = fun
    # domain = fun.bounds
    # disc_X = utils.grid(domain, samp_per_dim=7)
    # # either compute y_max over all points, or know it before hand. or None.
    # # for regret computing purposes only
    # maxx = torch.Tensor([[.20169, .15001, .47687, .27533, .31165, .6573]])
    # ymax = obj_fn(maxx)

    # USER: create objective fn in objectives.py
    encoding = 'onehot'
    obj = objectives.GB1(encoding)
    batch_size = 96
    # obj = objectives.Hartmann_3d

    # relevant params:
    xi = .1 # beta param for UCB
    # n_pseudorand_init = 384
    # budget = 96

    n_pseudorand_init = batch_size
    budget = 384 - n_pseudorand_init #budget includes MLDE evaluation at the end with 96 samples
    activation = 'lrelu'

    # make dir to hold tensors
    path = '/home/jyang4/repos/DK-BO/'
    subdir = 'results/test/'
    subdir = path + subdir
    os.makedirs(subdir, exist_ok=True)
    # so have record of all params
    os.system('cp ' + __file__ + ' ' + subdir)
    print('Script stored.')

    # USER: set # runs you wish to perform, and index them for saving
    runs = 1
    # start this at 0, -> however many runs you do total. i.e. 20
    index = 0

    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]

    # other params, don't worry about...
    acq_iter = None
    acqlr = .01
    train_iter = 100
    trainlr = .01
    n_rand_init = 0
    min_noise = 1e-6
    verbose = False
    epsilon = .01
    rand_restarts = 1000
    grid_size = None
    lengthscale_bounds = None
    # how often retrain? normally 1
    interval = 1
    # params to be removed; don't affect anything at this point
    noise = 0
    batch_size = 1
    num_fids = 1
    n_test = 1000 # rand case
    # necessary? remove?
    test_x, test_y = None, None #utils.samp_discrete(n_test, obj)

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')

    seeds = []
    with open('../rndseed.txt', 'r') as f:
        lines = f.readlines()
        for i in range(runs):
            print('run index: {}'.format(index+i))
            line = lines[i+index].strip('\n')
            print('seed: {}'.format(int(line)))
            seeds.append(int(line))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_simult_jobs = 10
    arg_list = []

    for r in range(index, index + runs):
        seed = seeds[r - index]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # TODO: switch to Sobol instead of uniform? this should be done for continuous fns, but not e.g. proteins
        def get_initial_points(dim, n_pts, seed=0):
            sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
            X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
            return X_init

        # start_x = utils.samp_discrete_X(n_pseudorand_init, disc_X)
        # print(obj_fn(start_x)[1], obj_fn(start_x)[1].shape)
        # start_y = obj_fn(start_x)[1]
        # start_x, start_y = utils.samp_discrete(n_pseudorand_init, obj)
        start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj)

        # do random search first
        if budget != 0:
            _, randy, rand_indices = utils.samp_discrete(budget + 96, obj)
            randy = torch.cat((start_y, randy), 0) #concatenate to the initial points
        # # do random search first
        # if budget != 0:
        #     randx = utils.samp_discrete_X(budget*batch_size, disc_X)
        #     randy = obj_fn(randx)[1]
        #     randy = torch.cat((start_y, randy), 0)
        else:
            randy = start_y
        temp = []
        for n in range(budget + 96 + 1):
            m = torch.max(randy[:n + n_pseudorand_init])
            reg = torch.reshape(torch.abs(ymax - m), (1, -1))
            temp.append(reg)
        tc = torch.cat(temp, 0)
        tc = torch.reshape(tc, (1, -1))
        torch.save(tc, subdir + 'Random_' + str(r + 1) + 'regret.pt')
        torch.save(randy, subdir + 'Random_' + str(r + 1) + 'y.pt')
        print('Random search done.')

        # USER: initialize each experiment.
        #   -- relevant params are: mtype, kernel, acq, architecture, (mc)dropout
        # TODO: this stuff should really get made into a dictionary instead of listing all out

        acq = 'botorch_ucb'

        # not meaningful for GP-BO
        dropout, mcdropout = 0, 0
        architecture = None

        ### MLDE models ###
        mtype = 'MLDE'

        #the first 2 args aren't used for MLDE
        # kernel = 'Lin'
        # acq = 'TS'
        # fname = mtype + '-greedy_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        ### GP models ###
        mtype = 'GP'

        # kernel = 'RBF'
        # acq = 'GREEDY'
        # fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # kernel = 'Lin'
        # acq = 'TS'
        # fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # kernel = 'RBF'
        # acq = 'TS'
        # fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # kernel = 'RBF'
        # acq = 'UCB'
        # fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # acq = 'EI'
        # fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        ### DK-BO models ###
        mtype = 'DKL'
        architecture = [domain[0].size(-1), 500, 150, 50]
        #architecture = [domain[0].shape[-1], 30, 1]
        # from dkl paper
        # architecture = [domain[0].size(-1), 1000, 500, 50, 2]

        # average models w/ default kernel, acq fn combinations
        # kernel = 'Lin'
        # acq = 'TS'
        # fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # mtype = 'BDKL'
        # kernel = 'Lin'
        # acq = 'UCB'
        # fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # kernel = 'RBF'
        # acq = 'GREEDY'
        # fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        kernel = 'RBF'
        acq = 'TS'
        fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        # kernel = 'RBF'
        # acq = 'UCB'
        # fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # acq = 'EI'
        # fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # MC dropout
        # dropout, mcdropout = 0.1, 0.1
        # kernel = 'Lin'
        # acq = 'TS'
        # fname = mtype + str(architecture[-1]) + 'MCD-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        # dropout, mcdropout = 0.1, 0.1
        # kernel = 'RBF'
        # acq = 'TS'
        # fname = mtype + str(architecture[-1]) + 'MCD-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        # p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, start_indices, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        # p.start()
        # print(fname)

        fname = 'DKL-Lin-TS-' + str(arc[1:-1]) + '_' + str(r + 1)
        args = BO_ARGS(
            # primary args
            mtype='DKL',
            kernel='lin',
            acq_fn='TS',
            # secondary args
            xi=.1,
            architecture=arc,
            activation='lrelu',
            min_noise=1e-6,
            trainlr=1e-2,
            train_iter=100,
            dropout=0,
            mcdropout=0,
            verbose=2,
            # usually don't change
            bb_fn=obj_fn,
            domain=domain,
            disc_X=disc_X,
            obj_max=ymax,
            noise_std=0,
            n_rand_init=0,
            budget=300,
            query_cost=1,
            queries_x=start_x,
            queries_y=start_y,
            savedir=subdir+fname,
        )
        arg_list.append((args, seed))

    total_time = time.time()
    with mp.Pool(10) as pool:
        pool.map(BayesianOptimization.run, arg_list)
        pool.close()
        pool.join()
        print(f'Total runtime: {time.time()-total_time}')
    print('Tensors will be saved in {}'.format(subdir))
