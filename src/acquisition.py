from __future__ import annotations

import math, time, copy
import random
import os

import gpytorch
import numpy as np
import torch
import botorch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples

import src.utils as utils

class Acquisition:
    """Generic class for acquisition functions that includes the function and
    its optimizer."""

    def __init__(self, acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir):
        """Initializes Acquisition object.
        Args:
            acq_fn_name: name of acquisition function ('GREEDY', 'UCB', or 'TS')
            domain: domain for acquisition function (only used if continuous design space)
            queries_x: already queried x values
            norm_y: already queried normalized y values
            normalizer: normalizer for y values
            disc_X: discrete domain for acquisition function
            verbose: verbosity level
            xi: parameter for UCB
            seed_index: index of seed
        """
        self.gpu = torch.cuda.is_available()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'

        self.acq = acq_fn_name
        self.queries_x = queries_x.double().to(self.device)
        self.nqueries = queries_x.shape[0]
        self.norm_y = norm_y.double()
        
        self.disc_X = disc_X.double()
        self.normalizer = normalizer
        self.verbose = verbose
        self.domain = domain # not used for a discrete domain
        self.xi = xi
        self.seed_index = seed_index
        self.save_dir = save_dir

        self.embeddings = None #embeddings from the neural network of the deep kernel
        self.preds = None #acquisition function values at each point in the discrete domain

    def get_next_query(self, samp_x, samp_y, samp_indices):
        """Returns the next sample to query."""

        self.preds[np.array(samp_indices, dtype=int)] = min(self.preds) #set the already queried values to the minumum of acquisition

        ind = np.argmax(self.preds)
        best_x = torch.reshape(self.disc_X[ind].detach(), (1, -1)).double()
        acq_val = self.preds[ind]
        best_idx = torch.tensor(ind)

        return best_x, acq_val, best_idx
    
    #TODO: write the methods that must be here 

class AcquisitionEnsemble(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, y_preds_full_all, normalizer, disc_X, verbose, xi, seed_index, save_dir):
        """
        Initializes Acquisition object for models that are ensembles (BOOSTING_ENSEMBLE or DNN_ENSEMBLE).
        Additionally takes in y_preds_full_all, the predictions of each model in the ensemble at each point in disc_X.
        """
        super().__init__(acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir)

        self.y_preds_full_all = y_preds_full_all

    def get_preds(self, X_pending):
        """
        Updates self.preds to be the acquisition function values at each point.
        X_pending are the previously queried points from the same batch, but is not used.
        """
        if self.acq.upper() == 'UCB':
            #alternatively could implement this as the best or second best value
            mu = torch.mean(self.y_preds_full_all, axis = 1)
            sigma = torch.std(self.y_preds_full_all, axis = 1)
            delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma
            torch.save(sigma*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'sigma.pt')
            torch.save(mu*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'mu.pt')
            self.preds = mu + delta
        elif self.acq.upper() == 'GREEDY':
            self.preds = torch.mean(self.y_preds_full_all, axis = 1)
        elif self.acq.upper() == 'TS':
            column = np.random.randint(self.y_preds_full_all.shape[1])
            self.preds = (self.y_preds_full_all[:, column])
        
        self.preds = self.preds.detach().numpy()

class AcquisitionGP(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, model, normalizer, disc_X, verbose, xi, seed_index, save_dir):
        """
        Initializes Acquisition object for models that are based on Gaussian processes (GP_BOTORCH or DKL_BOTORCH).
        Additionally takes in the trained model object.
        """
        super().__init__(acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir)

        self.model = model.double().to(self.device)

    def get_embedding(self):
        """
        For DKL_BOTORCH, passes  all encodings in disc_X through the trained neural network layers of the model. Only necessary for thompson sampling acquisition (TS)
        Updates self.embeddings to be the embeddings of each point in disc_X.
        """

        if self.model.dkl and self.acq.upper() == 'TS':
            self.embeddings = self.model.embed_batched_gpu(self.disc_X).double()
        else:
            self.embeddings = self.disc_X
       
    def get_preds(self, X_pending):
        """
        Passes the encoded values in disc_X through the acquisition function.
        Updates self.preds to be the acquisition function values at each point.
        X_pending are the previously queried points from the same batch, but is not used.
        """
        #Thompson Sampling
        if self.acq.upper() in ('TS'):
            model = copy.copy(self.model).to(self.device)
            #Deep Kernel
            if self.model.dkl:
                inputs = model.train_inputs[0].to(self.device)
                nn_x = model.embedding(inputs)
                model.train_inputs = (nn_x,)
            #GP
            else:
                model.train_inputs = (self.model.train_inputs[0],)
            
            #Sample a random function from the posterior
            gp_sample = get_gp_samples(
                    model=model,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
            )
            self.acquisition_function = PosteriorMean(model=gp_sample)
            
            self.preds = self.model.eval_acquisition_batched_gpu(self.embeddings, f=self.max_obj).cpu().detach().double()
        #For UCB and Greedy
        else: 
            if self.gpu: 
                self.model = self.model.cuda()
                with gpytorch.settings.fast_pred_var(), torch.no_grad():
                    mu, sigma = self.model.predict_batched_gpu(self.embeddings)
            else:
                ### TODO: need to fix this, currently does not work ###
                mu, sigma = self.model.predict(self.embeddings)

            if self.acq.upper() == 'UCB':
                delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma

                #save for uncertainty quantification
                torch.save(sigma*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'sigma.pt')
                torch.save(mu*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'mu.pt')
                self.preds = mu + delta
            elif self.acq.upper() == 'GREEDY':
                self.preds = mu.cpu()
        self.preds = self.preds.detach().numpy()

    def max_obj(self, x):
        """
        Acquisition function to maximize.
        """
        return self.acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])))