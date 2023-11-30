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

    def __init__(self, acq_fn_name, domain, queries_x, norm_y, disc_X, verbose, xi, seed_index, save_dir):
        """Initializes Acquisition object.
        -@param: acq_fn, takes in prior distribution and builds function
        -@param: next_query, optimizes acq_fn and returns next x val.
        """
        self.gpu = torch.cuda.is_available()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'

        self.acq = acq_fn_name
        self.queries_x = queries_x.double().to(self.device)
        self.norm_y = norm_y.double()
        
        self.disc_X = disc_X.double()
        self.verbose = verbose
        self.domain = domain # not used bc discrete domain
        self.xi = xi
        self.seed_index = seed_index
        self.save_dir = save_dir

        self.embeddings = None
        self.preds = None

    # get rid of extra params here and above
    def get_next_query(self, samp_x, samp_y):
        """Returns the next query input."""
        ind = torch.argmax(self.preds)
        best_x = torch.reshape(self.disc_X[ind].detach(), (1, -1)).double()
        acq_val = self.preds[ind].detach().double()
        #print("Best acq val" + str(acq_val))
        best_idx = ind
        
        # if maximizer already queried, take the "next best"
        if utils.find_x(best_x, samp_x.cpu()):
            #print('Best already taken, finding next best')
            best_x, acq_val, best_idx = utils.find_next_best(self.disc_X, self.preds, samp_x, samp_y)
            #print("Replacement acq val" + str(acq_val))

        return best_x, acq_val, best_idx
    
    #write the methods that must be here 

class AcquisitionEnsemble(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, y_preds_full_all, disc_X, verbose, xi, seed_index, save_dir):
        super().__init__(acq_fn_name, domain, queries_x, norm_y, disc_X, verbose, xi, seed_index, save_dir)
        self.y_preds_full_all = y_preds_full_all

    def get_preds(self, X_pending):

        #print (self.y_preds_full_all.shape)
        if self.acq.upper() == 'UCB':
            #could also just implement this as the best or second best value
            mu = torch.mean(self.y_preds_full_all, axis = 1)
            sigma = torch.std(self.y_preds_full_all, axis = 1)
            delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma
            torch.save(sigma, self.save_dir + 'sigma.pt')
            torch.save(mu, self.save_dir + 'mu.pt')
            self.preds = mu + delta
        elif self.acq.upper() == 'GREEDY':
            self.preds = torch.mean(self.y_preds_full_all, axis = 1)
        elif self.acq.upper() == 'EI':
            #how to calculate it in this case?
            improvements = self.y_preds_full_all - max(self.norm_y)
            #round to 0 if negative
            improvements[improvements < 0] = 0
            self.preds = torch.mean(improvements, axis = 1)
        elif self.acq.upper() == 'TS':
            #select a random moel
            column = np.random.randint(self.y_preds_full_all.shape[1])
            self.preds = (self.y_preds_full_all[:, column])
            #print(self.preds.shape)

class AcquisitionGP(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, model, disc_X, verbose, xi, seed_index, save_dir):
        super().__init__(acq_fn_name, domain, queries_x, norm_y, disc_X, verbose, xi, seed_index, save_dir)
        self.model = model.double().to(self.device)

    def get_embedding(self):
        """
        Embeds all of the values in disc_X using the deep NN layers, for TS acquisition with a neural network is used. Embedding is not changed for other acquisition functions.
        Updates self.embeddings to be the embeddings of each point in disc_X.
        """

        if self.model.dkl and self.acq.upper() == 'TS':
            # start= time.time()
            self.embeddings = self.model.embed_batched_gpu(self.disc_X).double()
            #print(os.getcwd())
            #torch.save(self.embeddings, 'embeddings.pt')
            # print('embedding time', time.time() - start)
        else:
            self.embeddings = self.disc_X
       
    def get_preds(self, X_pending):
        """
        Passes the encoded values in disc_X through the acquisition function.
        Updates self.preds to be the acquisition function values at each point.
        """
        #for botorch acquisition functions
        if self.acq.upper() in ('TS'):
            if self.model.lin and self.acq.upper() == 'TS':
                noise = self.model.get_kernel_noise().to(self.device).double()
                if self.model.dkl:
                    samp_x = samp_x #not sure what is going on heere, its a relic
                    # x is only nn embedding
                    nn_x = self.model.embedding(samp_x.double()).to(self.device)
                else:
                    nn_x = samp_x.double()#.to(self.device)
            else:
                #need to set self.train_inputs to the embedding, not the original
                model = copy.copy(self.model).to(self.device)
                if self.model.dkl:
                    inputs = model.train_inputs[0].to(self.device)
                    nn_x = model.embedding(inputs)
                    model.train_inputs = (nn_x,)
                else:
                    model.train_inputs = (self.model.train_inputs[0],)
                
                if self.acq.upper() == 'TS':
                    #only needs train inputs, train outputs, and covariance, likelihood
                    gp_sample = get_gp_samples(
                            model=model,
                            num_outputs=1,
                            n_samples=1,
                            num_rff_features=1000,
                    )
                    self.acquisition_function = PosteriorMean(model=gp_sample)
                
                self.preds = self.model.eval_acquisition_batched_gpu(self.embeddings, f=self.max_obj).cpu().detach().double()
        elif self.acq.upper() == 'QEI': 
            
            if X_pending is not None:
                X_pending = X_pending.to(self.device)
                print(X_pending.shape)

            sampler = botorch.sampling.SobolQMCNormalSampler(128)
            self.acquisition_function = botorch.acquisition.qNoisyExpectedImprovement(
                    model=self.model,
                    X_baseline=self.queries_x,
                    sampler=sampler.to(self.device),
                    prune_baseline=True,
                    # X_pending=None
                    X_pending=X_pending
                )
            #specificy sequential and say what was already appended
            #need to add an extra dimension to specify that each batch is a single point

            #shape is weird here, the embedding that is being passed into the acquisition function is gpu_batch x qbatch x d
            self.preds = self.model.eval_acquisition_batched_gpu(self.embeddings, f=self.max_obj).cpu().detach().double()
            
            # self.acquisition_function = botorch.acquisition.qNoisyExpectedImprovement(
            #         model=self.model,
            #         X_baseline=self.queries_x.cpu().detach(),
            #         sampler=sampler,
            #         prune_baseline=True,
            #         # X_pending=None
            #         X_pending=X_pending
            #     )
            # self.preds = self.acquisition_function.forward(self.embeddings.cpu().detach().double())
            # print(self.preds[10:])

        else: 
            if self.acq.upper() == 'EI':
                if self.gpu:
                    self.model = self.model.cuda()
                    #should samp_y should be updated with each query or not? For now, use the same maximum for every query before retraining the model
                    self.acquisition_function = botorch.acquisition.analytic.ExpectedImprovement(self.model, torch.max(self.norm_y))
                    self.preds = self.model.eval_acquisition_batched_gpu(self.embeddings, f=self.max_obj).cpu().detach().double()
            else:
                #UCB or Greedy
                if self.gpu: 
                    self.model = self.model.cuda()
                    # so don't put too much on gpu at once
                    with gpytorch.settings.fast_pred_var(), torch.no_grad():
                        mu, sigma = self.model.predict_batched_gpu(self.embeddings)
                else:
                    ### need to fix this, currently does not work ###
                    mu, sigma = self.model.predict(self.embeddings)

                if self.acq.upper() == 'UCB':
                    delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma
                    #save for uncertainty quantification
                    torch.save(sigma, self.save_dir + 'sigma.pt')
                    torch.save(mu, self.save_dir + 'mu.pt')
                    self.preds = mu + delta
                elif self.acq.upper() == 'GREEDY':
                    self.preds = mu.cpu()

    def max_obj(self, x):
        #add the extra dimension to specify qbatch is only 1
        #works with submodular optimization
        #return self.acquisition_function.forward(x)
        return self.acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])))


# USER: Acquisition function API:
#   Inputs: X (all possible candidates), samp_x, samp_y, gp model, OPT: xi (extra constant), batch size, verbose
#   Outputs: acq fn value on all X.

# def thompson_sampling(X, samp_x, samp_y, model, xi=None, batch=1000, verbose=2, embedded=False):
#     #start = time.time()
#     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     #self.device = 'cpu'

#     #converting the model to double is a naive fix, might be a better way to do this?
#     model, samp_x, samp_y = model.double().to(self.device), samp_x.to(self.device), samp_y.to(self.device)
#     # for linear kernel only
#     samp_y = torch.reshape(samp_y, (-1, 1))
#     # TODO: have model output this, since won't be the same across.
#     if model.lin:
#         noise = model.get_kernel_noise().to(self.device).double()
#         if model.dkl:
#             samp_x = samp_x
#             # x is only nn embedding
#             nn_x = model.embedding(samp_x.double()).to(self.device)
#         else:
#             nn_x = samp_x.double()#.to(self.device)
#     else:
#         # gp = gp.cpu()
#         #need to set self.train_inputs to the embedding, not the original
#         if model.dkl:
#             model = copy.copy(model).to(self.device)
#             inputs = model.train_inputs[0].to(self.device)
#             nn_x = model.embedding(inputs)
#             model.train_inputs = (nn_x,)

#             #line below doesn't seem to make a difference
#             #model.train_inputs = (model.embed_batched_gpu(inputs),)
#         else:
#             #is this the same as samp_x?
#             #nn_x = samp_x.double()

#             model.train_inputs = (model.train_inputs[0],)
#         #only needs train inputs, train outputs, and covariance, likelihood
#         gp_sample = get_gp_samples(
#                 model=model,
#                 num_outputs=1,
#                 n_samples=1,
#                 num_rff_features=1000,
#         )

#         acquisition_function = PosteriorMean(model=gp_sample)

#         def max_obj(x):
#             return acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])))
#             #return acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])).to(self.device))
        
#     if not embedded and model.dkl:
#         # start= time.time()
#         embeddings = model.embed_batched_gpu(X, batch_size=batch)
#         # print('embedding time', time.time() - start)
#     else:
#         embeddings = X

#     # start= time.time()
#     acq = model.eval_acquisition_batched_gpu(embeddings, batch_size=batch, f=max_obj)
#     # print('acquisition time', time.time() - start)

#     #print(time.time() - start)
#     return acq.cpu().double(), embeddings


# def upper_conf_bound(X, samp_x, samp_y, model, beta, batch=1000, verbose=2):
#     """
#     Computes UCB at points X, where beta represents exploration/exploitation tradeoff.
#     UCB(x) = mu(x) + sqrt(beta) * sigma(x)
#     """
#     #start = time.time()
#     if gpu: 
#         model = model.cuda()
#         # so don't put too much on gpu at once
#         with gpytorch.settings.fast_pred_var(), torch.no_grad():
#             mu, sigma = model.predict_batched_gpu(X, batch_size=batch)
#     else:
#         mu, sigma = model.predict(X)

#     delta = (beta * torch.ones_like(mu)).sqrt() * sigma
#     #print(time.time() - start)
#     return mu + delta

# def greedy(X, samp_x, samp_y, model, beta, batch=1000, verbose=2):
#     '''
#     Computes greedy acquisition function at points X
#     '''
#     #start = time.time()
#     if gpu: 
#         model = model.cuda()
#         # so don't put too much on gpu at once
#         with gpytorch.settings.fast_pred_var(), torch.no_grad():
#             mu, _ = model.predict_batched_gpu(X, batch_size=batch)
#     else:
#         mu, _ = model.predict(X)
#     #print(time.time() - start)
#     return mu.cpu()


# # TODO: batch not implemented here yet
# def expected_improvement(X, samp_x, samp_y, model, xi=None, batch=1000, verbose=2):
#     """
#     Computes the EI at points X.
#     EI(x) = E(max(y - best_f, 0)), y ~ f(x)
#     Args:
#         X: Points at which EI shall be computed (m x d).
#         X_sample: Sample locations (n x d).
#         Y_sample: Sample values (n x 1).
#         model: A GaussianProcessRegressor fitted to samples.
#         xi: Exploitation-exploration trade-off parameter.
#     Returns:
#         Expected improvements at points X.
#     """
    
#     #TODO: if not gpu
#     if gpu:
#         model = model.cuda()
#         samp_x = samp_x.cuda()
#     with gpytorch.settings.fast_pred_var(), torch.no_grad():
#         mu, sigma = model.predict_batched_gpu(X, batch_size=1000)
#         f_best_seen = torch.max(
#             model(samp_x).mean.cpu()
#         )  # not quite correct for noisy obs
#     impr = mu - f_best_seen  # - xi
#     Z = impr / sigma
#     normal = torch.distributions.Normal(torch.zeros_like(Z), torch.ones_like(Z))
#     cdf = normal.cdf(Z)
#     pdf = torch.exp(normal.log_prob(Z))
#     # exploitation term + exploration term
#     ei = impr * cdf + sigma * pdf
#     # det. set to 0--is this necessary?
#     ei[sigma == 0.0] = 0.0
#     return ei