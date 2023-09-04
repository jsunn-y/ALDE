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

from optimize import BayesianOptimization, BO_ARGS
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
    obj = objectives.FoldX
    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]

    n_pseudorand_init = 10
    budget = 300

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')
    path = './test/'
    subdir = path + 'results/paper_2022/botorch/bnn_test_fx/'
    os.makedirs(subdir, exist_ok=True)
    # so have record of all params
    os.system('cp ' + __file__ + ' ' + subdir)
    print('Script stored.')

    # TODO: set list of arcs you'd like to try:
    arc_list = [
       [domain[0].shape[-1], 30, 1],
    #    [domain[0].shape[-1], 50, 1],
    ]

    # USER: set # runs you wish to perform, and index them for saving
    runs = 10
    # start this at 0, -> however many runs you do total. i.e. 20
    index = 0
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

    # TODO: set this to how many exps can fit on GPU at once
    num_simult_jobs = 6
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

        start_x, start_y = utils.samp_discrete(n_pseudorand_init, obj)
        # do random search first
        if budget != 0:
            _, randy = utils.samp_discrete(budget, obj)
            randy = torch.cat((start_y, randy), 0)
        else:
            randy = start_y
        temp = []
        for n in range(budget + 1):
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

        # GP models

        fname = 'GPB-lin-TS' + '_' + str(r + 1)
        args = BO_ARGS(
            # primary args
            mtype='GP_BOTORCH',
            kernel='lin',
            acq_fn='TS',
            # secondary args
            xi=.1,
            architecture=[domain[0].size(-1), 1],
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

        fname = 'GPB-lin-qnEI' + '_' + str(r + 1)
        args = BO_ARGS(
            # primary args
            mtype='GP_BOTORCH',
            kernel='lin',
            acq_fn='BOTORCH_QNEI',
            # secondary args
            xi=.1,
            architecture=[domain[0].size(-1), 1],
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

        # DK-BO models
        mtype = 'DKL'

        for arc in arc_list:
            fname = 'DKLB-Lin-qnEI-' + str(arc[1:-1]) + '_' + str(r + 1)
            args = BO_ARGS(
                # primary args
                mtype='DKL_BOTORCH',
                kernel='lin',
                acq_fn='BOTORCH_QNEI',
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

    # launch!
    total_time = time.time()
    with mp.Pool(num_simult_jobs) as pool:
        pool.starmap(BayesianOptimization.run, arg_list)
        pool.close()
        pool.join()
    print(f'Total runtime: {time.time()-total_time}')

    print('Tensors will be saved in {}'.format(subdir))
