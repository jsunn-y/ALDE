from __future__ import annotations

import math, time, copy
import random

import gpytorch
import numpy as np
import torch
import botorch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples

import src.utils as utils


gpu = torch.cuda.is_available()
#gpu = False

class Acquisition:
    """Generic class for acquisition functions that includes the function and
    its optimizer. After initialization, call get_next_query() to obtain the
    suggested next query input."""

    def __init__(self, acq_fn, opt_fn, domain, xi):
        """Initializes Acquisition object.
        -@param: acq_fn, takes in prior distribution and builds function
        -@param: next_query, optimizes acq_fn and returns next x val.
        """
        self.acq_fn = acq_fn
        self.next_query = acq_optimize_discrete
        self.next_query = opt_fn
        self.domain = domain # not used bc discrete domain
        self.xi = xi

    # get rid of extra params here and above
    def get_next_query(
        self,
        samp_x,
        samp_y,
        model,
        disc_X,
        verbose=2,
        index=0,
        preds=None,
        embeddings=None
    ):
        """Gets next query by optimizing the acquisition function. Restarts to
        avoid local extrema. Minimization objective should be created inside the
        optimization function (though could be impl. outside?).
            -@param: domain, a tuple of (minx, maxx)
            -@param: samples, a tuple of (x, y) where both are torch tensors
            -@param: prior, a model (e.g. GP, deep kernel)
            -@param: likelihood (probably mll)
            -return: torch.tensor (same dim as x)
        """
        result = self.next_query(
            self.acq_fn,
            samp_x,
            samp_y,
            model,
            self.xi,
            disc_X=disc_X,
            verbose=verbose,
            index=index,
            preds=preds,
            embeddings=embeddings)
        return result


def acq_optimize_discrete(
    acq,
    samp_x,
    samp_y,
    model,
    xi,
    disc_X,
    batch_size=1000,
    verbose=2,
    index=0,
    preds=None,
    embeddings=None,
):
    """
    Proposes the next sampling point by optimizing the acquisition function.
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        model: A GaussianProcessRegressor or other model fitted to samples.
    Returns:
        Location of the acquisition function maximum.
    """
    # USER: choose an acq fn. to implement your own, see above interface.
    # to add another botorch acq, create a function wrapper as above.
    acq_dict = {
        "EI": expected_improvement,
        "UCB": upper_conf_bound,
        "TS": thompson_sampling,
        "GREEDY": greedy
    }
    if acq.upper() not in acq_dict:
        raise NotImplementedError(f"Acq fn not recognized/implemented. Choose one of {acq_dict.keys()} or add your own.")
    else:
        fn = acq_dict[acq.upper()]

    batch = disc_X.double()

    # predict and find acq. fn maximizer
    #for UCB don't always take the best one if the batch is greater than 1
    #don't actually need this anymore, cause it will automatically find the next best proposed point
    if (acq.upper() != 'TS') and index == 0:
        preds = fn(batch, samp_x, samp_y, model, xi, batch=batch_size, verbose=verbose).detach()
    elif (acq.upper() == 'TS') and index == 0:
        preds,  embeddings = fn(batch, samp_x, samp_y, model, xi, batch=batch_size, verbose=verbose, embedded = False)
        preds = preds.detach()
        embeddings = embeddings.detach()
    elif (acq.upper() == 'TS') and index != 0: # use the new embeddings
        preds,  embeddings = fn(embeddings, samp_x, samp_y, model, xi, batch=batch_size, verbose=verbose, embedded = True)
        preds = preds.detach()
        embeddings = embeddings.detach()
    else:
        pass
       
    ind = torch.argmax(preds)
    best_x = torch.reshape(batch[ind].detach(), (1, -1)).double()
    acq_val = preds[ind].detach().double()
    best_idx = ind
    
    # if maximizer already queried, take the "next best"
    if utils.find_x(best_x, samp_x.cpu()):
        best_x, acq_val, best_idx = utils.find_next_best(batch, preds, samp_x, samp_y)
    return best_x, acq_val, best_idx, preds, embeddings

# USER: Acquisition function API:
#   Inputs: X (all possible candidates), samp_x, samp_y, gp model, OPT: xi (extra constant), batch size, verbose
#   Outputs: acq fn value on all X.

def thompson_sampling(X, samp_x, samp_y, model, xi=None, batch=1000, verbose=2, embedded=False):
    #start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    #converting the model to double is a naive fix, might be a better way to do this?
    model, samp_x, samp_y = model.double().to(device), samp_x.to(device), samp_y.to(device)
    # for linear kernel only
    samp_y = torch.reshape(samp_y, (-1, 1))
    # TODO: have model output this, since won't be the same across.
    if model.lin:
        noise = model.get_kernel_noise().to(device).double()
        if model.dkl:
            samp_x = samp_x
            # x is only nn embedding
            nn_x = model.embedding(samp_x.double()).to(device)
        else:
            nn_x = samp_x.double()#.to(device)
    else:
        # gp = gp.cpu()
        #need to set self.train_inputs to the embedding, not the original
        if model.dkl:
            model = copy.copy(model).to(device)
            inputs = model.train_inputs[0].to(device)
            nn_x = model.embedding(inputs)
            model.train_inputs = (nn_x,)

            #line below doesn't seem to make a difference
            #model.train_inputs = (model.embed_batched_gpu(inputs),)
        else:
            #is this the same as samp_x?
            #nn_x = samp_x.double()

            model.train_inputs = (model.train_inputs[0],)
        #only needs train inputs, train outputs, and covariance, likelihood
        gp_sample = get_gp_samples(
                model=model,
                num_outputs=1,
                n_samples=1,
                num_rff_features=1000,
        )

        acquisition_function = PosteriorMean(model=gp_sample)

        def max_obj(x):
            return acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])).to(device))
        
    if not embedded and model.dkl:
        # start= time.time()
        embeddings = model.embed_batched_gpu(X, batch_size=batch)
        # print('embedding time', time.time() - start)
    else:
        embeddings = X

    # start= time.time()
    acq = model.eval_acquisition_batched_gpu(embeddings, batch_size=batch, f=max_obj)
    # print('acquisition time', time.time() - start)

    #print(time.time() - start)
    return acq.cpu().double(), embeddings


def upper_conf_bound(X, samp_x, samp_y, model, beta, batch=1000, verbose=2):
    """
    Computes UCB at points X, where beta represents exploration/exploitation tradeoff.
    UCB(x) = mu(x) + sqrt(beta) * sigma(x)
    """
    #start = time.time()
    if gpu: 
        model = model.cuda()
        # so don't put too much on gpu at once
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            mu, sigma = model.predict_batched_gpu(X, batch_size=batch)
    else:
        mu, sigma = model.predict(X)

    delta = (beta * torch.ones_like(mu)).sqrt() * sigma
    #print(time.time() - start)
    return mu + delta

def greedy(X, samp_x, samp_y, model, beta, batch=1000, verbose=2):
    '''
    Computes greedy acquisition function at points X
    '''
    #start = time.time()
    if gpu: 
        model = model.cuda()
        # so don't put too much on gpu at once
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            mu, _ = model.predict_batched_gpu(X, batch_size=batch)
    else:
        mu, _ = model.predict(X)
    #print(time.time() - start)
    return mu.cpu()


# TODO: batch not implemented here yet
def expected_improvement(X, samp_x, samp_y, model, xi=None, batch=1000, verbose=2):
    """
    Computes the EI at points X.
    EI(x) = E(max(y - best_f, 0)), y ~ f(x)
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        model: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    Returns:
        Expected improvements at points X.
    """
    
    #TODO: if not gpu
    if gpu:
        model = model.cuda()
        samp_x = samp_x.cuda()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        mu, sigma = model.predict_batched_gpu(X, batch_size=1000)
        f_best_seen = torch.max(
            model(samp_x).mean.cpu()
        )  # not quite correct for noisy obs
    impr = mu - f_best_seen  # - xi
    Z = impr / sigma
    normal = torch.distributions.Normal(torch.zeros_like(Z), torch.ones_like(Z))
    cdf = normal.cdf(Z)
    pdf = torch.exp(normal.log_prob(Z))
    # exploitation term + exploration term
    ei = impr * cdf + sigma * pdf
    # det. set to 0--is this necessary?
    ei[sigma == 0.0] = 0.0
    return ei

