# ### DEPRECATED, here for reference.


import numpy as np
import torch
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

'''
Sample experiment runner script for DK-BO. Launches optimization runs as
separate processes.
This uses the Hartmann 6D dataset as an example.
'''

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # set up objective func and related values
    obj = objectives.Hartmann_6d
    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]

    # fixed params throughout all experiments:
    budget = 300
    acq_iter = None
    acqlr = .01
    train_iter = 100
    trainlr = .01
    # beta param for UCB
    xi = .1
    n_pseudorand_init = 10
    n_rand_init = 0
    activation = 'lrelu'
    min_noise = None
    verbose = False
    epsilon = .01
    rand_restarts = 1000
    grid_size = None
    lengthscale_bounds = None
    # how often retrain? normally 1
    interval = 1
    # params to be removed; don't affect anything at this point
    noise = .01
    batch_size = 1
    num_fids = 1
    n_test = 1000 # rand case
    test_x, test_y = utils.samp_discrete(n_test, obj)
    ensemble = None

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')
    # make dir to hold tensors
    # path = '/scratch/ml/'
    # subdir = 'results/h6d_test'
    path = '/scratch/ml/jbowden/'
    subdir = 'results/paper_2022/h6d/'
    subdir = path + subdir
    os.system('mkdir ' + subdir)
    # so have record of all params
    os.system('cp ' + __file__ + ' ' + subdir)
    print('Script stored.')

    runs = 1
    # start this at 0, -> however many runs you do total. i.e. 20
    index = 9
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

    for r in range(index, index + runs):
        seed = seeds[r - index]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        start_x, start_y = utils.samp_discrete(n_pseudorand_init, obj)
        # do random search first
        if budget != 0:
            _, randy = utils.samp_discrete(budget*batch_size, obj)
            randy = torch.cat((start_y, randy), 0)
        else:
            randy = start_y
        temp = []
        for n in range(budget*batch_size + 1):
            m = torch.max(randy[:n + n_pseudorand_init])
            reg = torch.reshape(torch.abs(ymax - m), (1, -1))
            temp.append(reg)
        tc = torch.cat(temp, 0)
        tc = torch.reshape(tc, (1, -1))
        torch.save(tc, subdir + 'Random_' + str(r + 1) + 'regret.pt')
        torch.save(randy, subdir + 'Random_' + str(r + 1) + 'y.pt')
        print('Random search done.')

        # not meaningful for GP-BO
        dropout, mcdropout = 0, 0
        architecture = None

        # GP models
        mtype = 'GP'
        kernel = 'Lin'
        acq = 'TS'
        fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        # GP-Lin-UCB unstable

        kernel = 'RBF'
        acq = 'UCB'
        fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        acq = 'EI'
        fname = mtype + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        # DK-BO models
        mtype = 'DKL'
        architecture = [domain[0].size(-1), 500, 150, 50]
        # from dkl paper
        # architecture = [domain[0].size(-1), 1000, 500, 50, 2]

        # average models w/ default kernel, acq fn combinations
        kernel = 'Lin'
        acq = 'TS'
        fname = mtype + 'sweep-' + str(architecture[-1]) + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        acq = 'UCB'
        fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        kernel = 'RBF'
        fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        acq = 'EI'
        fname = mtype + 'sweep-' + str(architecture[-1]) + 'DO-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

        # MC dropout
        dropout, mcdropout = 0.1, 0.1
        kernel = 'Lin'
        acq = 'TS'
        fname = mtype + str(architecture[-1]) + 'MCD-' + str(dropout) + '-' + kernel + '-' + acq + '_' + str(r + 1)
        p = mp.Process(target=BayesianOptimization.run, args=(obj_fn, domain, acq, architecture, activation, n_rand_init, mtype, num_fids, dropout, mcdropout, ensemble, acq_iter, train_iter, xi, acqlr, trainlr, kernel, rand_restarts, budget, noise, min_noise, lengthscale_bounds, start_x, start_y, test_x, test_y, grid_size, disc_X, ymax, verbose, subdir + fname, batch_size, seed, interval))
        p.start()
        print(fname)

    print('Tensors will be saved in {}'.format(subdir))
