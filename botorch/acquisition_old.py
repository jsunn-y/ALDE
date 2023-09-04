import torch
import gpytorch
import botorch
import numpy as np
import os
import sys
import random
from datetime import datetime
import math
import warnings

import utils
import objectives


class Acquisition:
    '''Generic class for acquisition functions that includes the function and
    its optimizer. After initialization, call get_next_query() to obtain the
    suggested next query input.'''

    def __init__(self, acq_fn, opt_fn, domain, xi, lr, num_iter, num_restarts, num_fids=1):
        '''Initializes Acquisition object.
            -@param: acq_fn, takes in prior distribution and builds function
            -@param: next_query, optimizes acq_fn and returns next x val.
        '''
        self.acq_fn = acq_fn
        self.next_query = acq_optimize_discrete
        self.next_query = opt_fn
        self.domain = domain
        self.xi = xi
        self.lr = lr
        self.num_iter = num_iter
        self.num_restarts = num_restarts
        self.num_fids = num_fids

# get rid of extra params here and above
    def get_next_query(self, domain, samp_x, samp_y, model, disc_X, double=False, verbose=True, batch_size=1, mf=False, fid=None):
        '''Gets next query by optimizing the acquisition function. Restarts to
        avoid local extrema. Minimization objective should be created inside the
        optimization function (though could be impl. outside?).
            -@param: domain, a tuple of (minx, maxx)
            -@param: samples, a tuple of (x, y) where both are torch tensors
            -@param: prior, a model (e.g. GP, deep kernel)
            -@param: likelihood (probably mll)
            -return: torch.tensor (same dim as x)
        '''
        result = self.next_query(self.acq_fn, self.domain, samp_x, samp_y, model, self.xi, self.lr, self.num_iter, n_restarts=self.num_restarts, double=double, disc_X=disc_X, verbose=verbose, batch_size=batch_size)
        return result
# should just stick big fn here

gpu = torch.cuda.is_available()

# USER: Acquisition function API:
#   Inputs: X (all possible candidates), samp_x, samp_y, gp model, OPT: xi (extra constant), batch size
#   Outputs: acq fn value on all X.

def thompson_sampling(X, samp_x, samp_y, gp, xi=None, batch=1000):
    # for linear kernel only
    samp_y = torch.reshape(samp_y, (-1, 1)).double()
    noise_var = gp.likelihood.noise.cpu()
    if gp.dkl:
        if gpu:
            gp = gp.cuda()
            samp_x = samp_x.double().cuda()
            gp.eval()
        # x is only nn embedding
        nn_x = gp.embedding(samp_x).cpu()
    else:
        nn_x = samp_x.double().cpu()

    temp = torch.inverse(torch.mm(nn_x.t(), nn_x) + (noise_var * torch.eye(nn_x.size(-1))))
    mu_weights = torch.reshape(torch.mm(torch.mm(temp, nn_x.t()), samp_y), (1, -1))
    sigma_weights = temp * noise_var
    weights = torch.distributions.multivariate_normal.MultivariateNormal(mu_weights, sigma_weights)

    # # if already queried, re sample
    # next_x = None
    # count = 0
    # new_pts, acq_vals = None, None
    w = weights.sample()

    def max_obj(x):
        # could put more samples here and just add all up? or maybe just take worst as lower bound of sorts?
        res = torch.mm(x, w.t())
        return res

    # argmax
    pred = []
    k = X.size()[0]
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for n in range(0, k, batch):
            if n + batch > k:
                x = X[n:].double()
            else:
                x = X[n:n+batch].double()
            if gp.dkl:
                if gpu:
                    x = x.cuda()
                emb = gp.embedding(x).cpu()
            else:
                emb = x.cpu()
            acq = max_obj(emb)
            pred.append(acq)
        pred = torch.cat(pred, 0)
    return pred

def upper_conf_bound(X, samp_x, samp_y, gp, beta, batch=1000):
    '''
    Computes UCB at points X, where beta represents exploration/exploitation tradeoff.
    UCB(x) = mu(x) + sqrt(beta) * sigma(x)
    '''
    if gpu:
        gp = gp.cuda()
    gp.eval()
    k = X.size()[0]
    mu, sigma = [], []
    # so don't put too much on gpu at once
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for n in range(0, k, batch):
            if n + batch > k:
                x = X[n:]
            else:
                x = X[n:n+batch]
            if gpu: # do here so don't have to put whole set on gpu
                x = x.cuda()
            acq = gp(x)
            mu.append(acq.mean.cpu())
            sigma.append(acq.stddev.cpu())
        mu = torch.cat(mu, 0)
        sigma = torch.cat(sigma, 0)
    delta = (beta * torch.ones_like(mu)).sqrt() * sigma

    return mu.cpu() + delta.cpu()


# TODO: batch not implemented here yet
def expected_improvement(X, samp_x, samp_y, gp, xi=None, batch=1000):
    '''
    Computes the EI at points X.
    EI(x) = E(max(y - best_f, 0)), y ~ f(x)
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gp: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    Returns:
        Expected improvements at points X.
    '''
    # convert this over and test?
    with gpytorch.settings.fast_pred_var():
        pred = gp(X)
    mu = pred.mean
    sigma = pred.stddev
    mu_sample = gp(samp_x).mean
    mu_sample_opt = torch.max(mu_sample)
    # probably switch this errstate thing
    # with np.errstate(divide='warn'):
    imp = mu - mu_sample_opt #- xi
    Z = imp / sigma
    normal = torch.distributions.Normal(torch.zeros_like(Z), torch.ones_like(Z))
    cdf = normal.cdf(Z)
    pdf = torch.exp(normal.log_prob(Z))
    # exploitation term + exploration term
    ei = imp * cdf + sigma * pdf
    ei[sigma == 0.0] = 0.0
    return ei

# some botorch acq wrappers

def botorch_expected_improvement(X, samp_x, samp_y, gp, xi=None, batch=1000):
    fn = botorch.acquisition.analytic.ExpectedImprovement(gp, torch.max(samp_y))
    return fn(X.reshape(X.shape[0],1,X.shape[-1]))

def botorch_upper_conf_bound(X, samp_x, samp_y, gp, xi=None, batch=1000):
    fn = botorch.acquisition.analytic.UpperConfidenceBound(gp, xi)
    return fn(X.reshape(X.shape[0],1,X.shape[-1]))


def acq_optimize_discrete(acq, bounds, samp_x, samp_y, model, xi, lr, num_iter, disc_X, n_restarts=10, min_dist=.01, double=False, verbose=True, epsilon=.001, batch_size=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
    Returns:
        Location of the acquisition function maximum.
    '''
    gp = model.model
    # single, longer_args = True, False
    
    # USER: choose an acq fn. to implement your own, see above interface.
    # to add another botorch acq, create a function wrapper as above.
    acq_dict = {
        'EI': expected_improvement,
        'UCB': upper_conf_bound,
        'TS': thompson_sampling,
        'BOTORCH_EI': botorch_expected_improvement,
        'BOTORCH_UCB': botorch_upper_conf_bound,

    }
    if acq.upper() not in acq_dict:
        print('Acq fn not recognized/implemented.')
        print(acq_dict.keys())
    else:
        fn = acq_dict[acq.upper()]

    # if acq.upper() in ['EI', 'EXP', 'EXPIMP', 'IMP']:
    #     fn = expected_improvement
    # elif acq.upper() in ['UCB', 'UPPER', 'CONF']:
    #     fn = upper_conf_bound
    # elif acq.upper() in ['EG', 'EPSG', 'EPSGREEDY', 'EPSGR', 'EGREEDY']:
    #     fn = epsilon_greedy
    #     single = False # do on all datapoints
    # elif acq.upper() in ['TS', 'THOMPSON', 'SAMPLING']:
    #     fn = thompson_sampling
    #     # longer_args = True
    #     # opt = False # do on all datapoints
    # elif acq.upper() in ['BOTORCH_QEI']:
    #     fn = botorch.acquisition.qExpectedImprovement(model, train_Y.max(), maximize=True)

    
    batch = disc_X.float()
    if double:
        batch = batch.double()

    # if single == False: # not really used, just for epsilon greedy
    #     with torch.no_grad():
    #         next_x, acq_val = fn(batch, samp_x, samp_y, gp, epsilon)
    #         return next_x.float(), acq_val.float()
    # else: # actual acquisition used
    preds = fn(batch, samp_x, samp_y, gp, xi).detach()
    ind = torch.argmax(preds)
    best_x = torch.reshape(batch[ind].detach(), (1, -1)).float()
    acq_val = preds[ind].detach().float()

    if(utils.find_x(best_x, samp_x.cpu())):
        # if top value in array, then find top k values and check until find good one.
        # guaranteed to find something not queried yet. naive impl.
        preds = torch.reshape(preds, (1, -1))[0]
        np_pred = preds.numpy()
        k = samp_y.size(0) + 1
        # print("k {}".format(k))
        inds = np.argpartition(np_pred, -k)[-k:]
        top_pred = np_pred[inds]
        # sort ascending
        inds2 = np.argsort(top_pred)[::-1]
        sorted_inds = inds[inds2]
        # naive impl
        # start at 1 since we already tried the max above
        redund = 1
        while(redund < k):
            best_x = torch.reshape(batch[sorted_inds[redund]], (1, -1)).cpu().float()
            acq_val = torch.reshape(preds[sorted_inds[redund]], (1, 1)).cpu().float()
            if utils.find_x(best_x, samp_x.cpu()) == False:
                break
            else:
                redund += 1
        # print("redund: {}".format(redund))
    return best_x, acq_val






# ########## deprecated

# TODO: should be replaced w botorch continuous optimizer
# # does optimization on rand restarts one by one. takes a lot more time.
# # acq, bounds, samp_x, samp_y, model, xi, lr, num_iter, disc_X=None, n_restarts=10, min_dist=.01, double=False, verbose=True, epsilon=.001, batch_size=1
# def acq_optimize_continuous(acq_fn, bounds, samp_x, gp, xi, lr, num_iter, n_restarts=25, min_dist=.01, verbose=True, epsilon=.001):
#     '''
#     Proposes the next sampling point by optimizing the acquisition function.
#     Args:
#         acquisition: Acquisition function.
#         X_sample: Sample locations (n x d).
#         Y_sample: Sample values (n x 1).
#         gpr: A GaussianProcessRegressor fitted to samples.
#     Returns:
#         Location of the acquisition function maximum.
#     '''
#     def max_obj(X):
#         # Maximization objective is the acquisition function
#         return acq_fn(X, samp_x, gp, xi)

#     rand = []
#     for x in np.arange(n_restarts):
#         inp = utils.rand_samp(bounds).float()
#         # print(inp)
#         if torch.cuda.is_available():
#             inp = inp.cuda()
#         inp.requires_grad = True
#         acqoptimizer = torch.optim.Adam([{'params':inp, 'constraints':torch.distributions.constraints.interval(-1, 1)}], lr=lr)
#         prev = 0
#         diff = math.inf
#         iter = 0
#         if gpu:
#             gp = gp.cuda()
#             samp_x = samp_x.cuda()
#         while iter < num_iter:
#             # Zero backprop gradients
#             acqoptimizer.zero_grad()
#             # Calc loss and backprop derivatives
#             output = -max_obj(inp)
#             # lossfn is just acq value (want to maximize, so make negative)
#             acqloss = output
#             acqloss.backward()
#             acqoptimizer.step()
#             iter += 1
#             if abs(prev - acqloss.item()) < epsilon:
#                 # print('Iter %d - Loss: %.3f (converged)' % (iter, acqloss.item()))
#                 break
#             # make sure values are staying in domain
#             with torch.no_grad():
#                 # okay assuming will norm all dim to 0, 1
#                 inp = torch.clamp(inp, min=-1, max=1)
#         rand.append(inp)
#     batch = torch.cat(rand, 0).float()
#     del acqloss, output
#     ####################
#     vals = max_obj(batch)
#     if gpu:
#         gp = gp.cpu()
#         batch = batch.cpu()
#         vals = vals.cpu()
#         samp_x = samp_x.cpu()
#         torch.cuda.empty_cache()
#     best_x = batch[0]
#     best_val = vals[0]
#     for x in range(n_restarts):
#         if vals[x] > best_val:
#             best_x = batch[x]
#             best_val = vals[x]
#     best_x = torch.reshape(best_x, (1, best_x.size(0)))
#     return best_x

# def epsilon_greedy(X, samp_x, samp_y, gp, epsilon):
#     # select best value with 1-eps probability, and other rand value with eps/k probability
#     # need to be checking ALL datapoints
#     # call UCB to incl. sigma?
#     if 'torch' not in str(X.dtype):
#         X = torch.reshape(torch.from_numpy(X), (1,-1)).float()
#     k = X.size(0)
#     beta = .01

#     if gpu:
#         gp = gp.cuda()
#         X = X.cuda()

#     if random.random() < (1 - epsilon):
#         # take best
#         gp.eval()
#         preds = []
#         with gpytorch.settings.fast_pred_var():
#             for n in range(0, k, 100):
#                 if n + 100 > k:
#                     # batch = gp(X[n:]).mean
#                     mu, delta = upper_conf_bound(X[n:], samp_x, gp, beta)
#                 else:
#                     mu, delta = upper_conf_bound(X[n:n+100], samp_x, gp, beta)
#                     # batch = gp(X[n:n+100]).mean
#                 preds.append(mu+delta)
#         preds = torch.cat(preds, -1)
#         # print(preds)
#         ind = torch.argmax(preds)
#         acq_val = preds[ind]
#     else:
#         # draw randomly--should we make sure don't requery? low prob but if seeded maybe not?
#         next_x = None
#         count = 0
#         # while(utils.find_x(next_x, samp_x.cpu())):
#         print('rand')
#         ind = random.randint(0, k)

#         with gpytorch.settings.fast_pred_var():
#             acq_val = gp(torch.reshape(X[ind], (1, -1))).mean

#     # print(ind)
#     # print(X[ind])
#     # print(acq_val)
#     return X[ind].detach(), acq_val.detach()