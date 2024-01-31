from __future__ import annotations
import torch
import gpytorch
import botorch

import numpy as np
import pandas as pd
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
    encoding = 'onehot'
    df = pd.read_csv('/disk1/jyang4/repos/data/Pgb_fitness.csv')
    n_samples = len(df)
    obj_col = 'Diff'
    obj = objectives.Production(df, encoding, obj_col)

    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]
    disc_y = obj.get_points()[1]
    
    #number of proposals to screen in the next round
    batch_size = 96
    budget = batch_size

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')
    
    # make dir to hold tensors
    path = 'results/production/'
    subdir = path + 'round1/'

    #save the strings of combos for the search space
    np.save(path + "combos.npy", np.array(obj.all_combos))

    os.makedirs(subdir, exist_ok=True)
    os.system('cp ' + __file__ + ' ' + subdir)
    print('Script stored.')

    runs = 1 #only perform one prediction
    index = 0 #for reproducibility
    seeds = []

    with open('src/rndseed.txt', 'r') as f:
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

        kernel='RBF'
        for mtype in ['BOOSTING_ENSEMBLE', 'GP_BOTORCH', 'DNN_ENSEMBLE', 'DKL_BOTORCH',]: 
            for acq_fn in ['GREEDY', 'UCB', 'TS']: 
                dropout=0

                if mtype == 'GP_BOTORCH' and 'ESM2' in encoding:
                    lr = 1e-1
                else:
                    lr = 1e-3
                
                num_simult_jobs = 1

                if 'DNN' in mtype and 'ENSEMBLE' in mtype:
                    if 'onehot' in encoding:
                        arc  = [domain[0].size(-1), 50, 30, 1]
                    elif 'AA' in encoding:
                        arc  = [domain[0].size(-1), 12, 8, 1]
                    elif 'georgiev' in encoding:
                        arc  = [domain[0].size(-1), 50, 30, 1]
                    elif 'ESM2' in encoding:
                        arc  = [domain[0].size(-1), 500, 150, 50, 1] 
                elif 'GP' in mtype:
                    arc = [domain[0].size(-1), 1]
                elif 'DKL' in mtype:
                    if 'onehot' in encoding:
                        arc  = [domain[0].size(-1), 50, 30, 1]
                    elif 'AA' in encoding:
                        arc  = [domain[0].size(-1), 12, 8, 1]
                    elif 'georgiev' in encoding:
                        arc  = [domain[0].size(-1), 50, 30, 1]
                    else:
                        arc  = [domain[0].size(-1), 500, 150, 50, 1] 

                fname = mtype + '-DO-' + str(dropout) + '-' + kernel + '-' + acq_fn + '-' + str(arc[-2:]) + '_' + str(r + 1)

                args = BO_ARGS(
                    mtype=mtype,
                    kernel=kernel,
                    acq_fn=acq_fn,
                    xi=4,
                    architecture=arc,
                    activation='lrelu',
                    min_noise=1e-6,
                    trainlr=lr,
                    train_iter=300,
                    dropout=dropout,
                    mcdropout=0,
                    verbose=2,
                    bb_fn=obj_fn,
                    domain=domain,
                    disc_X=disc_X,
                    disc_y=disc_y,
                    noise_std=0,
                    n_rand_init=0,
                    budget=budget,
                    query_cost=1,
                    queries_x=obj.Xtrain,
                    queries_y=obj.ytrain,
                    indices=obj.train_indices,
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
