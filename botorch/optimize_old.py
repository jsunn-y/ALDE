import os
import random
from datetime import datetime
from typing import Literal

import botorch
import gpytorch
import numpy as np
import torch
from torch import Tensor

import acquisition
import models
from objectives import ObjectiveFunc
import utils

#### Generic Bayesian Optimization framework


class BayesianOptimization:
    '''Generic Bayesian Optimization class. While still budget left, optimizes
    given surrogate model (current support for GP, DK) and queries objective
    function.
    '''

    def __init__(self,
                 bb_func: ObjectiveFunc,
                 domain: tuple[Tensor, Tensor],
                 acq: Literal['EI', 'UCB', 'TS'],
                 architecture=None,
                 activation: str | None = None,
                 n_rand_init=5,
                 mtype: Literal['DKL', 'GP'] = 'DKL',
                 num_fids: int = 1,
                 dropout=0,
                 mcdropout=0,
                 acq_iter=20,
                 train_iter=30,
                 xi=.1,
                 epsilon=.01,
                 acqlr=.01,
                 trainlr=.01,
                 kernel='rbf',
                 rand_restarts: int = 10,
                 budget=100,
                 query_cost=1,
                 noise_std: float = .001,
                 min_noise=None,
                 lengthscale_bounds=None,
                 queries_x: Tensor | None = None,
                 queries_y: Tensor | None = None,
                 test_x=None,
                 test_y=None, grid_size=10):
        '''Initialize BayesOpt.

        Args:
            bb_func: the black box (objective) func, takes in x values and kwarg noise
            domain: (minx, maxx) tuple where each elem is Tensor of shape [d]
            acq: name of acquisition function
            architecture: TODO
            activation: name of nonlinear activation function
            n_rand_init: # random points to query at start
            mtype: name of BO model, one of ['GP', 'DKL']
            num_fids: # of fidelities
            rand_restarts: # of times to randomly restart optimization
            -@param: epsilon, diff to stop training at
            -@param: lr, learning rate
            -@param: kernel, the base kernel for the GP or deep kernel
            -@param: budget
            -@param: query_cost, cost of each query (will be needed for mf)
            noise_std: std for Gaussian noise
            queries_x: any prior inputs queried
            queries_y: any prior queries, corresponding to queries_x
        Assuming that the output is a single scalar for now.
        '''

        print("\nInitializing Bayesian Optimization.----------------------\n")
        # normalize domain and get reversion func, conversion func
        self.domain = domain

        # init existing samples and normalize inputs
        if queries_x is None or queries_y is None:
            assert self.queries_x is None
            assert self.queries_y is None
            self.queries_x = torch.empty(0)
            self.queries_y = torch.empty(0)
        else:
            self.queries_x = queries_x
            self.queries_y = queries_y
            print(f'Num prev. inputs: {queries_x.size(0)}')
            # normalize each init input
            self.queries_x = botorch.utils.transforms.normalize(self.queries_x, domain)

        # set up noise func
        def noise() -> float | Tensor:
            if noise_std == 0:
                return 0
            n = torch.normal(mean=torch.zeros(1), std=noise_std)
            return n
        self.noise = noise

        self.bb_func = bb_func
        # add rand
        if n_rand_init > 0:
            print(f'Initializing {n_rand_init} random samples.')
            rand_x, rand_y = utils.batch_rand_samp(n_rand_init, domain, self.bb_func, noise=self.noise)
            # normalize x after querying
            self.queries_x = torch.cat((self.queries_x, botorch.utils.transforms.normalize(rand_x, domain)), dim=0)
            self.queries_y = torch.cat((self.queries_y, rand_y), dim=0)
            print('Used {n_rand_init}/{budget} \n')

        # init rest of variables
        self.mtype = mtype
        self.reset = (mtype != 'GP')  # don't reset if GP
        self.budget = budget
        self.cost = 0
        self.query_cost = query_cost
        self.train_iter = train_iter
        self.architecture = architecture
        self.activation = activation
        self.kernel = kernel
        self.epsilon = epsilon
        self.trainlr = trainlr
        self.grid_size = grid_size
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.min_noise = min_noise
        self.dropout, self.mcdropout = dropout, mcdropout
        self.num_fids = num_fids
        self.arc_fn = (lambda arc, n, budget: arc) # ID fn as default.

        # TODO: cont, discr obj selection
        self.acq = acquisition.Acquisition(
            acq, acquisition.acq_optimize_discrete, self.domain, xi, acqlr,
            acq_iter, rand_restarts, num_fids=self.num_fids)

        self.test_x, self.test_y = None, None

        print("Initialization completed.\n")

    # make savedir req
    def optimize(self,
                 disc_X: Tensor | None = None,
                 obj_max=None,
                 verbose=True,
                 savedir='',
                 batch_size=1,
                 interval=10) -> tuple:
        '''Main loop of bayesian optimization. Runs until budget is exhausted.

        Args:
            disc_X: list of discrete X inputs
            obj_max: float, max of obj_fn if known for evaluation purposes

        Returns: final posterior model, a MVN
        '''

        def save_tensors():
            print('Saving: {}'.format(savedir))
            if obj_max is not None:
                try:
                    torch.save(self.regret.cpu(), savedir + 'regret.pt')
                    print('Regret saved.')
                except Exception as e:
                    print(e)
            try:
                torch.save(self.queries_y.cpu(), savedir + 'y.pt')
                print('Y values saved.')
                # torch.save(self.trainmae, savedir + 'trainmae.pt')
                # torch.save(self.ratrainmae, savedir + 'ratrainmae.pt')
                # print('Train MAE saved.')
                # t = torch.cat(self.preds, -1)
                # torch.save(t, savedir + 'acq.pt')
                # print('Acq. values saved.')
            except Exception as e:
                print(e)
            try:
                torch.save(self.queries_x.cpu(), savedir + 'x_norm.pt')
                print('norm X values saved.')
            except Exception as e:
                print(e)
            try:
                botorch.utils.transforms.unnormalize(self.queries_x.cpu(), self.domain)
                torch.save(botorch.utils.transforms.unnormalize(self.queries_x.cpu(), self.domain), savedir + 'x.pt')
                print('X values saved.')
            except Exception as e:
                print(e)
            # try:
            #     t = torch.from_numpy(np.array(self.lcs))
            #     torch.save(t, savedir + 'lc.pt')
            #     print('Learning curves saved.')
            # except Exception as e:
            #     print(e)
            # try:
            #     t = torch.cat(self.losses, 0).cpu()
            #     # print(t)
            #     # t = torch.from_numpy(np.array(self.losses))
            #     torch.save(t, savedir + 'll.pt')
            #     print('Log likelihoods saved.')
            # except Exception as e:
            #     print(e)
            # if self.test_x!= None and self.test_y is not None:
            #     try:
            #         torch.save(self.testmae, savedir + 'testmae.pt')
            #         torch.save(self.ratestmae, savedir + 'ratestmae.pt')
            #         print('Test MAE saved.')
            #     except Exception as e:
            #         print(e)
            # try:
            #     t = torch.cat(self.errors, 0).cpu()
            #     # print(t)
            #     # t = torch.from_numpy(np.array(self.losses))
            #     torch.save(t, savedir + 'errors.pt')
            #     print('Errs saved.')
            # except Exception as e:
            #     print(e)

        print("Beginning optimization, {}.-----------------\n".format(os.path.basename(savedir)))
        start = datetime.now()
        # normalize full x set for certain acq fns
        if disc_X is not None:
            self.disc_X = botorch.utils.transforms.normalize(disc_X, self.domain)
        # get normalization factor--obj max for test cases and exp. max otherwise
        self.max = torch.max(self.queries_y)
        self.normalizer = torch.max(torch.abs(self.queries_y))
        self.preds, self.lcs, self.losses, self.errors = [], [], [], []
        # set max, regret, trainmae, testmae from init queries, assuming >0
        if obj_max is not None:
            simp_reg = torch.reshape(torch.abs(obj_max - self.max), (1,1))
            self.regret = simp_reg
            if verbose:
                print("Regret: {}".format(simp_reg))
        self.norm_y = self.queries_y / self.normalizer
        # self.double = self.model.kernel.lower()=='lin'

        # train init model on random samples w/ norm y
        print("Creating initial prior.")
        self.model = models.Model(
            self.queries_x,
            torch.reshape(self.norm_y, (1, -1))[0],
            self.min_noise,
            self.train_iter,
            savedir,
            mtype=self.mtype,
            kernel=self.kernel,
            architecture=self.arc_fn(self.architecture, self.queries_y.size(0), self.budget),
            dropout=self.dropout,
            mcdropout=self.mcdropout,
            activation=self.activation,
            # epsilon=self.epsilon,
            lr=self.trainlr,
            verbose=verbose)
        _, ll, _, _ = self.model.train(self.queries_x, torch.reshape(self.norm_y, (1, -1))[0], reset=False)
        self.losses.append(ll)
        if self.mtype.upper() != 'ENSEMBLE':
            self.double = self.kernel.lower()=='lin'
        else:
            self.double = self.model.kernel.lower()=='lin'
        # took this out for now bc MAE not particularly helpful metric and takes a lot of compute
        # trainmae = utils.calc_mae(self.queries_x, torch.reshape(self.norm_y, (1, -1))[0], self.model.model, double=self.double)
        # log.metric('trainmae', trainmae.item())
        # if verbose:
        #     print('Train MAE: {}'.format(trainmae))
        # self.trainmae = torch.tensor([trainmae])
        # self.ratrainmae = torch.mean(self.trainmae, 0, keepdim=True)
        # if self.test_x!= None and self.test_y is not None:
        #     testmae = utils.calc_mae(self.test_x, torch.reshape(self.test_y, (1, -1))[0] / self.normalizer, self.model.model, double=self.double)
        #     # log.metric('testmae', testmae.item())
        #     if verbose:
        #         print('Test MAE: {}'.format(testmae))
        #     self.testmae = torch.tensor([testmae])
        #     self.ratestmae = torch.mean(self.testmae, 0, keepdim=True)

        # main loop; form and optimize acq_func to find next query x based on model (prior)
        while self.cost < self.budget:
            
            ###this function could return the whole batch (UCB) or a for loop (sampling)
            x, ypred = self.acq.get_next_query(self.domain, self.queries_x, self.norm_y, self.model.model, disc_X=self.disc_X, double=self.double, verbose=verbose, batch_size=batch_size, fid=0)

            #update cost more
            with torch.no_grad():
                x_ind = torch.reshape(x[0], (1, -1))
                y = self.bb_func(botorch.utils.transforms.unnormalize(x_ind, self.domain), noise=self.noise())
                if len(y) == 2:
                    _, y = y
                y = torch.reshape(y, (1, 1))[0]
            self.queries_x = torch.cat((self.queries_x, x_ind.float()), dim=0)
            self.queries_y = torch.cat((self.queries_y, y.float()), dim=0)
            self.preds.append(ypred * self.normalizer)
            if verbose:
                print("x: {}, y: {}".format(x_ind, y[0]))
            if self.cost%interval == 0:
                self.max = torch.max(self.queries_y)
                self.normalizer = torch.max(torch.abs(self.queries_y))

            # update normalized output tensor
            self.norm_y = self.queries_y / self.normalizer

            # update regr eval
            if obj_max is not None:
                simp_reg = torch.reshape(torch.abs(obj_max - self.max), (1,1))
                self.regret = torch.cat((self.regret, simp_reg), -1)

                #changed this so that updates are printed every 24 samples
                if self.cost%(24) == 0:
                    print(f"{os.path.basename(savedir)} | Regret: {simp_reg.item():.4f} | Used {self.cost}/{self.budget} budget.\n\n")
            # trainmae = utils.calc_mae(self.queries_x, torch.reshape(self.norm_y, (1, -1))[0], self.model.model, double=self.double)
            # # log.metric('trainmae', trainmae.item())
            # if verbose or self.cost%(self.budget/10) == 0:
            #     print('Train MAE: {}'.format(trainmae))
            # self.trainmae = torch.cat((self.trainmae, torch.tensor([trainmae])), 0)
            # self.ratrainmae = torch.cat((self.ratrainmae, torch.mean(self.trainmae, 0, keepdim=True)), 0)
            # if self.test_x!= None and self.test_y is not None:
            #     testmae = utils.calc_mae(self.test_x, torch.reshape(self.test_y, (1, -1))[0] / self.normalizer, self.model.model, double=self.double)
            #     # log.metric('testmae', testmae.item())
            #     if verbose or self.cost%(self.budget/10) == 0:
            #         print('Test MAE: {}'.format(testmae))
            #     self.testmae = torch.cat((self.testmae, torch.tensor([testmae])), 0)
            #     self.ratestmae = torch.cat((self.ratestmae, torch.mean(self.testmae, 0, keepdim=True)), 0)

            # add cost
            self.cost += self.query_cost
            # print("Used {}/{} budget.\n".format(self.cost, self.budget))
            # track progress at intervals
            if savedir is not None and self.cost%(24) == 0:
                save_tensors()
            # retraining intervals
            if self.cost%interval == 0 and self.cost != self.budget:
                _, ll, _, _ = self.model.train(self.queries_x, self.norm_y, reset=self.reset, dynamic_arc=self.arc_fn(self.architecture, self.queries_y.size(0), self.budget))
                # self.losses.append(ll)

        print(f"{os.path.basename(savedir)} | Optimization runtime: {datetime.now() - start} | Regret: {simp_reg.item():.4f}")
        if savedir is not None:
            save_tensors()
        return self.model, self.normalizer

    @staticmethod
    def run(bb_func, domain, acq, architecture, activation, n_rand_init, mtype,
            num_fids, dropout, mcdropout, acq_iter, train_iter, xi,
            acqlr, trainlr, kernel, rand_restarts, budget, noise_std, min_noise,
            lengthscale_bounds, queries_x, queries_y, test_x, test_y, grid_size,
            disc_X: Tensor | None,
            obj_max, verbose, savedir, batch_size, seed, interval):
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        test = BayesianOptimization(
            bb_func, domain, acq, architecture=architecture,
            activation=activation, n_rand_init=n_rand_init, mtype=mtype,
            num_fids=num_fids, dropout=dropout, mcdropout=mcdropout,
            acq_iter=acq_iter, train_iter=train_iter, xi=xi,
            acqlr=acqlr, trainlr=trainlr, kernel=kernel,
            rand_restarts=rand_restarts, budget=budget, noise_std=noise_std,
            min_noise=min_noise, lengthscale_bounds=lengthscale_bounds,
            queries_x=queries_x, queries_y=queries_y, test_x=test_x,
            test_y=test_y, grid_size=grid_size)
        return test.optimize(disc_X=disc_X, obj_max=obj_max, verbose=verbose, savedir=savedir, batch_size=batch_size, interval=interval)
