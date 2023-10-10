from __future__ import annotations

from collections.abc import Sequence, Mapping
from datetime import datetime
import os, time, sys
import random
from typing import Literal, Callable
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

#### BO struct ####

@dataclass
class BO_ARGS(MapClass):
    """General model class args. Each model class can
    take some subset of these as arguments; others discarded."""
    # dataset
    bb_fn: utils.ObjectiveFunc = None
    domain: tuple[Tensor, Tensor] = None
    disc_X: Tensor = None
    obj_max: Tensor | float = None
    noise_std: utils.Noise = 0 # generally set 0 unless want to inject noise
    n_rand_init: int = 0 # if you want it to auto query more pts
    batch_size: int = 1 # for batched BO
    queries_x: Tensor | None = None
    queries_y: Tensor | None = None
    indices: Tensor | None = None
    # model
    mtype: Literal['DKL', 'GP'] = 'DKL'
    kernel: str = 'rbf'
    architecture: Sequence[int] = None
    activation: str = 'relu'
    min_noise: float = 1e-6
    trainlr: float = .01
    train_iter: int = 100
    dropout: float = 0.0
    mcdropout: float = 0.0
    # acquisition
    acq_fn: str = None # name change
    xi: float =.1 # make this more transparent
    # BO
    budget: int = 300
    query_cost: float = 1.
    savedir: str = 'results/'
    verbose: int = 1 # 0, 1, 2, 3
    run_mlde: bool = True
    

#### Generic Bayesian Optimization framework
class BayesianOptimization:
    '''Generic Bayesian Optimization class. While still budget left, optimizes
    given surrogate model (current support for GP, DK) and queries objective
    function.
    '''

    def __init__(self,
                 bb_fn: utils.ObjectiveFunc,
                 domain: tuple[Tensor, Tensor],
                 disc_X: Tensor | None = None,
                 obj_max=None,
                 acq_fn: Literal['EI', 'UCB', 'TS'] | None = None,
                 architecture: Sequence[int] | None = None,
                 activation: str | None = None,
                 n_rand_init: int = 5,
                 mtype: Literal['DKL', 'CDKL', 'GP'] = 'DKL',
                 dropout=0,
                 mcdropout=0,
                 train_iter=100,
                 xi=.1,
                 trainlr=.01,
                 kernel='rbf',
                 budget=100,
                 batch_size=1,
                 query_cost=1,
                 noise_std: float = 0.0,
                 min_noise=None,
                 queries_x: Tensor | None = None,
                 queries_y: Tensor | None = None,
                 indices: Tensor | None = None,
                 savedir='',
                 verbose=True,
                 run_mlde=True,
                 *_, **__):
        '''Initialize BayesOpt.

        Args:
            bb_fn: the black box (objective) func, takes in x values and kwarg noise
            domain: (minx, maxx) tuple where each elem is Tensor of shape [d]
            acq_fn: name of acquisition function
            architecture: sizes of layers in NN, inc. input and output dims,
                set None if not using DKL
            activation: name of nonlinear activation function, set None if not using DKL
            n_rand_init: # random points to query at start
            mtype: name of BO model, one of ['GP', 'DKL']
            -@param: lr, learning rate
            -@param: kernel, the base kernel for the GP or deep kernel
            -@param: budget
            -@param: query_cost, cost of each query (will be needed for mf)
            noise_std: std for Gaussian noise
            min_noise: GP likelihood noise constraint
            queries_x: any prior inputs queried
            queries_y: any prior queries, corresponding to queries_x
        Assuming that the output is a single scalar for now.
        '''

        print("\nInitializing Bayesian Optimization.----------------------\n")
        # normalize domain and get reversion func, conversion func
        self.domain = domain
        # normalize full x set for certain acq fns
        if disc_X is not None:
            self.disc_X = botorch.utils.transforms.normalize(disc_X, self.domain)
            #self.disc_X = disc_X
        self.obj_max = obj_max
        self.verbose = verbose
        self.acq_fn = acq_fn
        self.run_mlde = run_mlde
        self.xi = xi

        # init existing samples and normalize inputs
        if queries_x is None or queries_y is None:
            assert queries_x is None
            assert queries_y is None
            self.queries_x = torch.empty(0)
            self.queries_y = torch.empty(0)
        else:
            self.queries_x = queries_x
            self.queries_y = queries_y
            print(f'Num prev. inputs: {queries_x.size(0)}')
            # normalize each init input
            self.queries_x = botorch.utils.transforms.normalize(self.queries_x, domain)
        self.indices = indices

        # set up noise func
        def noise() -> float | Tensor:
            if noise_std == 0:
                return 0
            n = torch.normal(mean=torch.zeros(1), std=noise_std)
            return n
        self.noise = noise

        self.bb_fn = bb_fn
        # add rand
        if n_rand_init > 0:
            if self.verbose >= 1: print(f'Initializing {n_rand_init} random samples.')
            # Pretty sure this doesn't work anymore, need to use discrete_sample but don't have obj object
            rand_x, rand_y = utils.batch_rand_samp(n_rand_init, domain, self.bb_fn, noise=self.noise)
            # normalize x after querying
            self.queries_x = torch.cat((self.queries_x, botorch.utils.transforms.normalize(rand_x, domain)), dim=0)
            self.queries_y = torch.cat((self.queries_y, rand_y), dim=0)
            if self.verbose >= 1: print('Used {n_rand_init}/{budget} \n')

        # init rest of variables
        self.mtype = mtype
        # TODO: fix this in models.py; for now, true for all non bayesian models
        self.reset = True #(mtype != 'GP')  # don't reset if GP
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
        self.arc_fn = (lambda arc, n, budget: arc) # ID fn as default.
        self.savedir = savedir
        

        # TODO: cont, discr obj selection
        # add multiple acquisition functions if desired
        if self.verbose >= 1: print("Initialization completed.\n")

    def optimize(self,) -> tuple:
        '''Main loop of bayesian optimization. Runs until budget is exhausted.

        Args:
            disc_X: list of discrete X inputs
            obj_max: double, max of obj_fn if known for evaluation purposes

        Returns: final posterior model, a MVN
        '''

        def save_tensors():
            print('Saving: {}'.format(self.savedir))
            if self.obj_max is not None:
                torch.save(self.regret.cpu(), self.savedir + 'regret.pt')
                if self.verbose >= 1: print('Regret saved.')
            torch.save(self.queries_y.cpu(), self.savedir + 'y.pt')
            if self.verbose >= 1: print('Y values saved.')
            torch.save(self.indices.cpu(), self.savedir + 'indices.pt')
            if self.verbose >= 1: print('Indices saved.')
            botorch.utils.transforms.unnormalize(self.queries_x.cpu(), self.domain)
            torch.save(botorch.utils.transforms.unnormalize(self.queries_x.cpu(), self.domain), self.savedir + 'x.pt')
            #torch.save(self.queries_x.cpu(), self.savedir + 'x.pt')
            if self.verbose >= 1: print('X values saved.')

        def load_tensors():
            if self.verbose >= 1: print('Loading in from: {}'.format(self.savedir))
            try:
                if self.obj_max is not None:
                    self.regret = torch.load(self.savedir + 'regret.pt')
                    if self.verbose >= 1: print('Regret loaded.')
                self.queries_y = torch.load(self.savedir + 'y.pt')
                if self.verbose >= 1: print('Y values loaded.')
                self.indices = torch.load(self.savedir + 'indices.pt')
                if self.verbose >= 1: print('Indices loaded.')
                self.queries_x = botorch.utils.transforms.normalize(torch.load(self.savedir + 'x.pt'), self.domain)
                if self.verbose >= 1: print('X values loaded.')
                self.cost = self.regret.shape[-1] - 1 # b/c takes regret at start before queries
            except Exception as e:
                if self.verbose >= 1: print(f'Could not load tensors: {e}')
            
        if self.verbose >= 1: print("Beginning optimization, {}.-----------------\n".format(os.path.basename(self.savedir)))
        start = datetime.now()

        #don't do this for now
        #load_tensors() # resume progress if possible

        # get normalization factor--obj max for test cases and exp. max otherwise
        self.max = torch.max(self.queries_y)
        self.normalizer = torch.max(torch.abs(self.queries_y))
        self.preds, self.lcs, self.losses, self.errors = [], [], [], []
        # set max, regret, trainmae, testmae from init queries, assuming >0
        if self.obj_max is not None:
            simp_reg = torch.reshape(torch.abs(self.obj_max - self.max), (1,1))
            self.regret = simp_reg
            max = torch.reshape(self.max/self.obj_max, (1,1))

            if self.verbose >= 2: print(f"\n{os.path.basename(self.savedir)} | Max: {max.item():.4f} | Regret: {simp_reg.item():.4f} | Used {self.cost}/{self.budget} budget.\n\n")
        self.norm_y = self.queries_y / self.normalizer #normalized y_queries

        # train init model on random samples w/ norm y
        print("Creating initial prior.")
        
        if self.mtype == 'MLDE':
            x_list, ypred_list, idx_list = self.train_predict_mlde_lite(self.queries_x, self.norm_y, self.batch_size)
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
            
            #start = time.time()
            _ = self.surrogate.train(self.queries_x, torch.reshape(self.norm_y, (1, -1))[0], reset=False)
            #print('Train time: ', time.time()-start)

        # main loop; form and optimize acq_func to find next query x based on model (prior)
        while self.cost < self.budget:
            
            for index in range(self.batch_size):

                #loop through each of the models and acquisition function types
                if self.mtype == 'MLDE':
                    #need to update this to produce the correct index
                    if self.batch_size == 1:
                        x = x_list.reshape(1,-1)
                        ypred = ypred_list
                        idx = idx_list
                    else:
                        #check the dimension of x here
                        x = x_list[index].reshape(1,-1)
                        ypred = ypred_list[index]
                        idx = idx_list[index]
                else:
                    #start = time.time()
                    if index == 0:
                        acq = acquisition.Acquisition(self.acq_fn, self.domain, self.queries_x, self.norm_y, self.surrogate.model, disc_X=self.disc_X, verbose=self.verbose, xi = self.xi)
                        acq.get_embedding()
                        acq.get_preds()
                    else:
                        if self.acq_fn == 'TS':
                            acq.get_preds()
                    
                    x, ypred, idx = acq.get_next_query(self.queries_x, self.norm_y)

                        #x, ypred, idx, preds, embeddings = self.acq.get_next_query(self.queries_x, self.norm_y, self.surrogate.model, disc_X=self.disc_X, verbose=self.verbose, index=index, preds=None, embeddings=None)

                    # elif self.acq_fn != 'TS':
                        #x, ypred, idx, preds, embeddings = self.acq.get_next_query(self.queries_x, self.norm_y, self.surrogate.model, disc_X=self.disc_X, verbose=self.verbose, index=index, preds=preds, embeddings=None)
                    # else:
                        #x, ypred, idx, preds, embeddings = self.acq.get_next_query(self.queries_x, self.norm_y, self.surrogate.model, disc_X=self.disc_X, verbose=self.verbose, index=index, preds=None, embeddings=embeddings)

                    #print('Evaluation time: ', time.time()-start)

                max, simp_reg = self.update_trajectory(x, ypred, idx)

            # track progress at intervals
            # only save progress outside of the batch training
            if self.savedir is not None and self.cost%(24) == 0:
                save_tensors()

            if self.cost < self.budget:
                if self.mtype == 'MLDE':
                    x_list, ypred_list, idx_list = self.train_predict_mlde_lite(self.queries_x, self.norm_y, self.batch_size)
                else:
                    _ = self.surrogate.train(self.queries_x, self.norm_y, reset=self.reset, dynamic_arc=self.arc_fn(self.architecture, self.queries_y.size(0), self.budget))
            
            #don't stop early
            # if simp_reg == 0:
            #     final_reg = torch.zeros(1,(self.budget-self.cost))
            #     self.regret = torch.cat((self.regret,final_reg),-1)
            #     if self.verbose >= 3: print(f'Regret is 0, terminating early at {self.cost}/{self.budget} budget.')
            #     break
        
        #do MLDE at the end
        #if self.cost < self.budget + 96:
        if self.run_mlde:
            print('Running MLDE for final 96 queries.')
            x_list, ypred_list, idx_list = self.train_predict_mlde_lite(self.queries_x, self.norm_y)
            #check the dimension of x here
            for x, ypred, idx in zip(x_list, ypred_list, idx_list):
                max, simp_reg = self.update_trajectory(x.reshape(1,-1), ypred, idx)

        
        if self.savedir is not None:
                save_tensors()

        if self.verbose >= 1: print(f"{os.path.basename(self.savedir)} | Optimization runtime: {datetime.now() - start} | Max: {max.item():.4f} | Regret: {simp_reg.item():.4f}")
        
        return

    @staticmethod
    def run(kwargs, seed):
    # def run(arg):
        # kwargs, seed = arg
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
    
    def update_trajectory(self, x, ypred, idx):
        with torch.no_grad():
            x_ind = torch.reshape(x[0], (1, -1))
            y = self.bb_fn(botorch.utils.transforms.unnormalize(x_ind, self.domain), noise=self.noise())
            #y = self.bb_fn(x_ind, noise=self.noise())
            if len(y) == 2: y = y[-1]
            y = torch.reshape(y, (1, 1))[0]
            idx = torch.reshape(idx, (1, 1))[0]
        
        #print(x_ind.shape)
        self.queries_x = torch.cat((self.queries_x, x_ind.double()), dim=0)
        self.queries_y = torch.cat((self.queries_y, y.double()), dim=0)
        self.indices = torch.cat((self.indices, idx.double()), dim=0)
        self.preds.append(ypred * self.normalizer)
        if self.verbose >= 3: print("x index: {}, y: {}".format(idx[0], y[0]))

        self.max = torch.max(self.queries_y)
        self.normalizer = torch.max(torch.abs(self.queries_y))
        # update normalized y tensor
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
        

    def train_predict_mlde_lite(self, X_train, y_train, num_preds=96):
        """
        Simplified training and prediction loop for MLDE. Always uses greedy acquisition.
        """
        
        #remove the already queried pts from disc_X (is this slower than james other way?)
        # train_indices = []
        # for row in X_train:
        #     train_indices.append(np.where((self.disc_X == row).all(axis=1))[0])
        train_indices = self.indices.numpy().astype(int)

        candidate_X = np.delete(self.disc_X, train_indices, 0)
        all_indices = np.arange(self.disc_X.shape[0])
        candidate_indices = np.delete(all_indices, train_indices, 0)
        y_preds_all = np.zeros((candidate_X.shape[0], 5))

        for i in range(5):
            X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=i)

            model_kwargs = {
            "objective": "reg:tweedie",
            "early_stopping_rounds": 10,
            "nthread": -1
            }
            clf = xgb.XGBRegressor(**model_kwargs)
            eval_set = [(X_validation, y_validation)]
            clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            y_preds = clf.predict(candidate_X)
            y_preds_all[:, i] = y_preds
        y_preds = np.mean(y_preds_all, axis = 1)

        top_candidate_indices = torch.topk(torch.tensor(y_preds), num_preds).indices
        top_X = candidate_X[top_candidate_indices]
        top_y_preds = y_preds[top_candidate_indices]
        top_indices = torch.tensor(candidate_indices[top_candidate_indices])

        return top_X, top_y_preds, top_indices
    
    

    

