from __future__ import annotations
import torch
import gpytorch
import botorch

import numpy as np
import random
# from datetime import datetime
# import glob
import os, time
# import math
import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor
import warnings

from src.optimize import BayesianOptimization, BO_ARGS
import src.objectives as objectives
import src.utils as utils

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
    encoding = 'GB1_ESM2'
    obj = objectives.Combo(encoding)

    #obj = objectives.Hartmann_6d()
    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]
    batch_size = 96

    n_pseudorand_init = batch_size
    budget = 384 - n_pseudorand_init #budget does not include MLDE evaluation at the end with 96 samples, and does not include random samples at the beginning

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')
    
    # make dir to hold tensors
    path = '/home/jyang4/repos/DKBO-MLDE/'
    subdir = path + 'results/GB1_ESM2/'
    #subdir = path + 'results/Hartmann_6d/'
    os.makedirs(subdir, exist_ok=True)
    # so have record of all params
    os.system('cp ' + __file__ + ' ' + subdir)
    print('Script stored.')

    # USER: set # runs you wish to perform, and index them for saving
    runs = 24
    # start this at 0, -> however many runs you do total. i.e. 20
    index = 0
    seeds = []

    with open('rndseed.txt', 'r') as f:
        lines = f.readlines()
        for i in range(runs):
            print('run index: {}'.format(index+i))
            line = lines[i+index].strip('\n')
            print('seed: {}'.format(int(line)))
            seeds.append(int(line))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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

        start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)

        # do random search first
        if budget != 0:
            _, randy, rand_indices = utils.samp_discrete(budget + 96, obj, seed)
            randy = torch.cat((start_y, randy), 0) #concatenate to the initial points
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

        kernel='RBF'
        for mtype in  ['GP_BOTORCH', 'DKL_BOTORCH', 'CDKL_BOTORCH']:
            for acq_fn in ['UCB', 'TS']: #'QEI', 'UCB','TS'
                dropout=0

                # if mtype == 'DKL' and acq_fn == 'TS' and "onehot" not in encoding:
                #     num_simult_jobs = 4 #current bottleneck is the maximum number of jobs that can fit on gpu memory
                # else:
                #     num_simult_jobs = 10
                num_simult_jobs = 1

                #last layer of architecture should be repeated, this gets fed to the GP
                if 'GP' in mtype:
                    arc = [domain[0].size(-1), 1] #use this architecture for GP
                elif 'CDKL' in mtype:
                    if 'onehot' in encoding:
                        #arc  = [int(domain[0].size(-1)/20), 20, 32, 32, 32, 64, 64]
                        arc  = [int(domain[0].size(-1)/20), 20, 32, 32, 32, 32, 32, 32]
                    elif 'ESM2' in encoding:
                        arc  = [int(domain[0].size(-1)/1280), 1280, 80, 40, 40, 32, 32, 32]
                elif 'DKL' in mtype:
                    if 'onehot' in encoding:
                        arc  = [domain[0].size(-1), 40, 20, 10, 10]
                    else:
                        arc  = [domain[0].size(-1), 500, 150, 50, 50] #becomes DKL automatically if more than two layers
                    # if 'ESM2' in encoding:
                    #     arc  = [int(domain[0].size(-1)/1280), 20, 16, 16, 16, 32, 32]
                else:
                    arc = [domain[0].size(-1), 1] #filler architecture for MLDE

                #fname = mtype + '-DO-' + str(dropout) + '-' + kernel + '-' + acq_fn + '_' + str(r + 1) + str(arc[1:-1]) + '_' + str(r + 1)
                if 'MLDE' in mtype:
                    fname = mtype +  '_' + str(r + 1)
                else:
                    fname = mtype + '-DO-' + str(dropout) + '-' + kernel + '-' + acq_fn + '-' + str(arc[-2:]) + '_' + str(r + 1)
                args = BO_ARGS(
                    # primary args
                    mtype=mtype,
                    kernel=kernel,
                    acq_fn=acq_fn,
                    # secondary args
                    xi=4,
                    architecture=arc,
                    activation='lrelu',
                    min_noise=1e-6,
                    trainlr=1e-3, #originally 1e-2 in james
                    train_iter=300,
                    dropout=dropout,
                    mcdropout=0,
                    verbose=2,
                    # usually don't change
                    bb_fn=obj_fn,
                    domain=domain,
                    disc_X=disc_X,
                    obj_max=ymax,
                    noise_std=0,
                    n_rand_init=0,
                    budget=budget,
                    query_cost=1,
                    queries_x=start_x,
                    queries_y=start_y,
                    indices=start_indices,
                    savedir=subdir+fname,
                    batch_size = batch_size
                )
                arg_list.append((args, seed))

    total_time = time.time()
    with mp.Pool(num_simult_jobs) as pool:
        pool.starmap(BayesianOptimization.run, arg_list)
        pool.close()
        pool.join()
        print(f'Total runtime: {time.time()-total_time}')
    
    print('Tensors will be saved in {}'.format(subdir))
