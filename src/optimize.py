from __future__ import annotations

from collections.abc import Sequence, Mapping
from datetime import datetime
import os, time, sys
import random
from typing import Literal
from dataclasses import dataclass, astuple

import botorch
import gpytorch
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
from torch import Tensor

import src.acquisition as acquisition
import src.models as models
import src.utils as utils
from src.utils import MapClass

@dataclass
class BO_ARGS(MapClass):
    """General model class args. Each model class can
    take some subset of these as arguments; others discarded."""
    bb_fn: utils.ObjectiveFunc = None
    domain: tuple[Tensor, Tensor] = None
    disc_X: Tensor = None
    disc_y: Tensor = None
    noise_std: utils.Noise = 0 # generally set 0 unless want to inject noise
    n_rand_init: int = 0 # if you want it to auto query more pts
    batch_size: int = 1 # for batched BO
    queries_x: Tensor | None = None
    queries_y: Tensor | None = None
    indices: Tensor | None = None
    # model
    mtype: Literal['DNN_ENSEMBLE', 'BOOSTING_ENSEMBLE', 'DKL_BOTORCH', 'GP_BOTORCH'] = 'DNN_ENSEMBLE'
    kernel: str = 'rbf'
    architecture: Sequence[int] = None
    activation: str = 'relu'
    min_noise: float = 1e-6
    trainlr: float = .01
    train_iter: int = 100
    dropout: float = 0.0
    mcdropout: float = 0.0
    acq_fn: str = None 
    xi: float = 4
    budget: int = 384
    query_cost: float = 1.
    savedir: str = 'results/'
    verbose: int = 1 # 0, 1, 2, 3
    seed_index: int = 0
    n_splits: int = 5
    bootstrap_size: float = 0.9

### Generic Bayesian Optimization framework ###
class BayesianOptimization:
    '''Generic Bayesian Optimization class. Simulates active learning (or Bayesian optimization) on a black box function.
    '''
    def __init__(self,
                 bb_fn: utils.ObjectiveFunc,
                 domain: tuple[Tensor, Tensor],
                 disc_X: Tensor | None = None,
                 disc_y = Tensor,
                 acq_fn: Literal['EI', 'UCB', 'TS'] | None = None,
                 architecture: Sequence[int] | None = None,
                 activation: str | None = None,
                 n_rand_init: int = 0,
                 mtype: Literal['DNN_ENSEMBLE', 'BOOSTING_ENSEMBLE', 'DKL_BOTORCH', 'GP_BOTORCH'] = 'DNN_ENSEMBLE',
                 dropout=0,
                 mcdropout=0,
                 train_iter=300,
                 xi=4,
                 trainlr=1e-3,
                 kernel='rbf',
                 budget=384,
                 batch_size=96,
                 query_cost=1,
                 noise_std: float = 0.0,
                 min_noise=None,
                 queries_x: Tensor | None = None,
                 queries_y: Tensor | None = None,
                 indices: Tensor | None = None,
                 savedir='',
                 seed_index=0,
                 verbose=True,
                 n_splits=5,
                 bootstrap_size=0.9,
                 *_, **__):
        '''Initialize BO.
        Args:
            bb_fn: black box function to optimize
            domain: tuple of (lower, upper) bounds for each input dimension
            disc_X: discrete X design space
            disc_y: discrete y design space
            acq_fn: name of acquisition function to use
            architecture: list of hidden layer sizes
            activation: activation function for neural network
            n_rand_init: number of random samples to further initialize
            mtype: one of ['GP_BOTORCH', 'DKL_BOTORCH', 'DNN_ENSEMBLE', 'BOOSTING_ENSEMBLE']
            dropout: training dropout for neural network
            mcdropout: test time dropout for neural network
            train_iter: number of training iterations
            xi: xi term for UCB
            trainlr: learning rate
            kernel: kernel for GP or DKL
            budget: number of queries to obtain
            batch_size: number of queries to obtain after each model is trained
            query_cost: cost of each query
            noise_std: noise
            min_noise: minimum noise constraint for GP
            queries_x: initial x inputs from random initialization
            queries_y: initial y inputs from random initialization
            indices: initial indices of the inputs, based on the full dataset
            savedir: directory to save results
            seed_index: random seed
            verbose: int btwn 0, 3 inclusive
            n_splits: number of splits to use for ensemble models
            bootstrap_size: size of the bootstrap sample for ensemble models
        '''

        print("\nInitializing Bayesian Optimization.----------------------\n")
        self.seed_index = seed_index
        self.domain = domain
        # normalize encoding to be between 0 and 1
        if disc_X is not None:
            self.disc_X = botorch.utils.transforms.normalize(disc_X, self.domain)
        self.disc_y = disc_y
        self.obj_max = torch.max(disc_y).double()
        self.verbose = verbose
        self.acq_fn = acq_fn
        self.xi = xi
        self.n_splits = n_splits
        self.bootstrap_size = bootstrap_size

        #init existing samples and normalize inputs
        if queries_x is None or queries_y is None:
            assert queries_x is None
            assert queries_y is None
            self.queries_x = torch.empty(0)
            self.queries_y = torch.empty(0)
        else:
            self.queries_x = queries_x #keeps track of the encodings for queried samples
            self.queries_y = queries_y #keeps track of the labels for queried samples
            print(f'Num prev. inputs: {queries_x.size(0)}')
            self.queries_x = botorch.utils.transforms.normalize(self.queries_x, domain)
        self.indices = indices #keeps track of the indices for queried samples, based on the compelte dataset

        # set up noise func
        def noise() -> float | Tensor:
            if noise_std == 0:
                return 0
            n = torch.normal(mean=torch.zeros(1), std=noise_std)
            return n
        self.noise = noise

        self.bb_fn = bb_fn

        # init rest of variables
        self.mtype = mtype
        self.reset = True 
        self.budget = budget
        self.batch_size = batch_size
        self.cost = 0
        self.query_cost = query_cost
        self.train_iter = train_iter
        self.architecture = architecture
        self.activation = activation
        self.kernel = kernel
        self.trainlr = trainlr
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.min_noise = min_noise
        self.dropout, self.mcdropout = dropout, mcdropout
        self.arc_fn = (lambda arc, n, budget: arc)
        self.savedir = savedir

        if self.verbose >= 1: print("Initialization completed.\n")

    def optimize(self,) -> tuple:
        """
        Main loop of bayesian optimization. Runs until budget is exhausted.
        """

        def save_tensors():
            """
            Save informatino about the queries, including the regret, y values, and indices.
            """
            print('Saving: {}'.format(self.savedir))
            # if self.obj_max is not None:
            #     torch.save(self.regret.cpu(), self.savedir + 'regret.pt')
            #     if self.verbose >= 1: print('Regret saved.')
            # torch.save(self.queries_y.cpu(), self.savedir + 'y.pt')
            # if self.verbose >= 1: print('Y values saved.')
            torch.save(self.indices.cpu(), self.savedir + 'indices.pt')
            if self.verbose >= 1: print('Indices saved.')
            
        if self.verbose >= 1: print("Beginning optimization, {}.-----------------\n".format(os.path.basename(self.savedir)))
        start = datetime.now()

        #normalize to the max y of the training data
        self.max = torch.max(self.queries_y)
        self.normalizer = torch.max(self.queries_y)
        self.preds, self.lcs, self.losses, self.errors = [], [], [], []

        # set max, regret, trainmae, testmae from init queries, assuming >0
        if self.obj_max is not None:
            simp_reg = torch.reshape(torch.abs(self.obj_max - self.max), (1,1))
            self.regret = simp_reg
            max = torch.reshape(self.max/self.obj_max, (1,1))

            if self.verbose >= 2: print(f"\n{os.path.basename(self.savedir)} | Max: {max.item():.4f} | Regret: {simp_reg.item():.4f} | Used {self.cost}/{self.budget} budget.\n\n")
        self.norm_y = self.queries_y / self.normalizer #normalized y_queries

        # train init model on random samples w/ norm y
        if 'BOOSTING' in self.mtype:
            if torch.cuda.is_available():
                    tree_method = 'gpu_hist'
            else:
                tree_method = 'hist'

            #use tweedie if all labels are positive
            if min(self.disc_y) >= 0:
                self.model_kwargs = {
                'tree_method': tree_method,
                "objective": "reg:tweedie",
                "early_stopping_rounds": 10,
                "nthread": -1
                }
            else:
                self.model_kwargs = {
                'tree_method': tree_method,
                "early_stopping_rounds": 10,
                "nthread": -1
                }
        else:
            self.surrogate = models.Model(
                self.queries_x,
                torch.reshape(self.norm_y, (1, -1))[0],
                self.min_noise,
                self.train_iter,
                self.savedir,
                mtype=self.mtype,
                kernel=self.kernel,
                architecture=self.arc_fn(self.architecture, self.queries_y.size(0), self.budget),
                dropout=self.dropout,
                mcdropout=self.mcdropout,
                activation=self.activation,
                lr=self.trainlr,
                verbose=self.verbose)
        
        # main loop; form and optimize acq_func to find next query x based on model
        while self.cost < self.budget:
            
            #ensemble models
            if 'ENSEMBLE' in self.mtype:
                y_preds_full_all = self.train_predict_ensemble(self.queries_x, self.norm_y)
            #GP models
            else:
                _ = self.surrogate.train(self.queries_x, torch.reshape(self.norm_y, (1, -1))[0], reset=True)

            for self.index in range(self.batch_size):
                if self.index == 0:
                    if 'ENSEMBLE' in self.mtype:
                        acq = acquisition.AcquisitionEnsemble(self.acq_fn, self.domain, self.queries_x, self.norm_y, y_preds_full_all, self.normalizer, disc_X=self.disc_X, verbose=self.verbose, xi = self.xi, seed_index = self.seed_index, save_dir = self.savedir)
                    else:
                        acq = acquisition.AcquisitionGP(self.acq_fn, self.domain, self.queries_x, self.norm_y, self.surrogate.model, self.normalizer, disc_X=self.disc_X, verbose=self.verbose, xi = self.xi, seed_index = self.seed_index, save_dir = self.savedir)
                        acq.get_embedding()
                    acq.get_preds(None)
                else:
                    #Thompson sampling requires a new acquisition function each time
                    if self.acq_fn == 'TS':
                        acq.get_preds(self.X_pending)
                
                x, acq_val, idx = acq.get_next_query(self.queries_x, self.norm_y, self.indices)
                max, simp_reg = self.update_trajectory(x, acq_val, idx)

            # track progress at intervals
            if self.savedir is not None and self.cost%(96) == 0:
                save_tensors()

        if self.savedir is not None:
            save_tensors()

        if self.verbose >= 1: print(f"{os.path.basename(self.savedir)} | Optimization runtime: {datetime.now() - start} | Max: {max.item():.4f} | Regret: {simp_reg.item():.4f}")
        
        return
    
    def update_trajectory(self, x, acq_val, idx):
        """
        Update the trajectory of the optimization.
        x: new point to query
        acq_val: acquisition function value at x
        idx: index of x in the discrete domain
        """
        with torch.no_grad():
            x_ind = torch.reshape(x[0], (1, -1))

            ## For finding the closest point in the discrete domain (slow)
            # y = self.bb_fn(botorch.utils.transforms.unnormalize(x_ind, self.domain), noise=self.noise())
            # #y = self.bb_fn(x_ind, noise=self.noise())
            # if len(y) == 2: y = y[-1]

            y = self.disc_y[idx]

            y = torch.reshape(y, (1, 1))[0]
            idx = torch.reshape(idx, (1, 1))[0]
        
        self.queries_x = torch.cat((self.queries_x, x_ind.double()), dim=0)
        self.X_pending = self.queries_x[-self.index-1:] #new points from the batch
        self.queries_y = torch.cat((self.queries_y, y.double()), dim=0)
        self.indices = torch.cat((self.indices, idx.double()), dim=0)
        
        if self.verbose >= 3: print("x index: {}, y: {}".format(idx[0], y[0]))

        self.max = torch.max(self.queries_y)
        self.normalizer = torch.max(self.queries_y)
        self.norm_y = self.queries_y / self.normalizer
        self.cost += self.query_cost

        # update regr eval
        if self.obj_max is not None:
            simp_reg = torch.reshape(torch.abs(self.obj_max - self.max), (1,1))
            self.regret = torch.cat((self.regret, simp_reg), -1)
            max = torch.reshape(self.max/self.obj_max, (1,1))

            if self.cost%(24) == 0:
                if self.verbose >= 2: print(f"\n{os.path.basename(self.savedir)} | Max: {max.item():.4f} | Regret: {simp_reg.item():.4f} | Used {self.cost}/{self.budget} budget.\n\n")

        return max, simp_reg
        
    def train_predict_ensemble(self, X_train_all, y_train_all):
        """
        Training and prediction loop for ensemble models.
        X_train_all: all of the training data
        y_train_all: all of the training labels
        bootstrap: whether to bootstrap the training data during ensembling
        """
        y_preds_full_all = np.zeros((self.disc_X.shape[0], self.n_splits))
        bootstrap = self.bootstrap_size < 1
        for i in range(self.n_splits):
            if bootstrap == True:
                #split for training and validation
                X_train, X_validation, y_train, y_validation = train_test_split(X_train_all, y_train_all, test_size=1-self.bootstrap_size, random_state=self.seed_index + i)
            else:
                X_train = X_train_all
                y_train = y_train_all
                X_validation = None
                y_validation = None
            
            if 'BOOSTING' in self.mtype:
                assert bootstrap == True
                clf = xgb.XGBRegressor(**self.model_kwargs)
                eval_set = [(X_validation, y_validation)]
                clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                y_preds_full = clf.predict(self.disc_X)
            elif 'DNN' in self.mtype:
                _ = self.surrogate.train(X_train, torch.reshape(y_train, (-1, 1)), reset=True)
                y_preds_full = self.surrogate.model(self.disc_X.to(self.surrogate.model.device)).detach().cpu().numpy().reshape(-1)

            y_preds_full_all[:, i] = y_preds_full

        return torch.tensor(y_preds_full_all)
    
    @staticmethod
    def run(kwargs, seed):
        print(f'Now launching {kwargs.savedir.split("/")[-1]}')
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        test = BayesianOptimization(**kwargs)
        res = test.optimize()
        return res
    
    

    

