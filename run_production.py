from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import random
import os, time
import multiprocessing as mp
import warnings
from src.optimize import BayesianOptimization, BO_ARGS
import src.objectives as objectives
import src.utils as utils

'''
Script for predicting a batch of sequences to use in the next round of active learning.
'''

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    protein = 'ParPgb' #name of the project
    encoding = 'onehot' #name of the project and the encoding to use
    df = pd.read_csv('data/' + protein + '/fitness.csv') #path to a csv file with sequence data and associated fitness values
    obj_col = 'Diff' #name of the fitness column to optimize
    obj = objectives.Production(df, protein, encoding, obj_col)

    # make dir to hold tensors
    path = 'results/ParPgb_production/'
    subdir = path + 'round1/'

    n_samples = len(df)
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
                else:
                    arc = [domain[0].size(-1), 1]

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
