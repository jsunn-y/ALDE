from __future__ import annotations

from collections.abc import Callable, Sequence, Mapping
from typing import Literal
from dataclasses import dataclass

import math
import os
import random

import gpytorch
import numpy as np
import scipy.optimize
import torch
from torch import Tensor

gpu = torch.cuda.is_available()

Noise = Tensor #| Tensor | Callable[..., float | Tensor]
ObjectiveFunc = Callable[[Tensor], Tensor]#Callable[[Tensor, Noise], Tensor]

@dataclass
class MapClass(Mapping):
    """Base struct that allows inheriting classes to be
    unpacked as kwargs using `**`."""

    def __getitem__(self, x):
        return self.__dict__[x]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

def f_batched_gpu(X, batch_size=1000, f=(lambda x: x), device='cuda'):
    with torch.no_grad():
        res = []
        for n in range(0, X.shape[0], batch_size):
            b = X[n : n + batch_size].to(device)
            res.append(f(b).to('cpu'))
            del b; torch.cuda.empty_cache()
        return torch.cat(res, 0)

# for acq
# def find_next_best(batch, preds, samp_x, samp_y):
#     # if top value in array, then find top k values and check until find good one.
#     # guaranteed to find something not queried yet. naive impl.
#     preds = torch.reshape(preds, (1, -1))[0]
#     np_pred = preds.numpy()
#     k = samp_y.size(0) + 1
#     # print("k {}".format(k))
#     inds = np.argpartition(preds.numpy(), -k)[-k:]
#     top_pred = np_pred[inds]
#     # sort ascending
#     inds2 = np.argsort(top_pred)[::-1]
#     sorted_inds = inds[inds2]
#     # naive impl
#     # start at 1 since we already tried the max above
#     redund = 1
#     while redund < k:
#         best_x = torch.reshape(batch[sorted_inds[redund]], (1, -1)).cpu().double()
#         acq_val = torch.reshape(preds[sorted_inds[redund]], (1, 1)).cpu().double()
#         if find_x(best_x, samp_x.cpu()) == False:
#             break
#         else:
#             redund += 1
#     # print("redund: {}".format(redund))
#     best_idx = torch.tensor(sorted_inds[redund])
#     return best_x, acq_val, best_idx


def rand_samp(domain: tuple[Tensor, Tensor]) -> Tensor:
    '''Generates a random sample in given domain.

    TODO: vectorize this

    Args:
        domain: (minx, maxx), where each is a Tensor of shape [d]

    Returns: Tensor of shape [1, d]
    '''
    xrange = domain[1] - domain[0]
    rand_x = torch.empty(1, xrange.size(-1))
    ind = 0
    if xrange.size(-1) == 1:
        rand_x[0] = np.random.rand()*xrange[0] + domain[0][0]
        return rand_x
    for dim in xrange[0]:
        rand_x[0][ind] = np.random.rand()*dim + domain[0][0][ind]
        ind += 1
    return rand_x


def batch_rand_samp(num_samp: int,
                    domain: tuple[Tensor, Tensor],
                    obj_fn: ObjectiveFunc,
                    noise: Noise = 0.,
                    verbose: bool = True
                    ) -> tuple[Tensor, Tensor]:
    """TODO: vectorize this"""
    if num_samp == 0:
        return torch.empty(0), torch.empty(0)
    lx, ly = [], []
    for _ in range(num_samp):
        rand_x = rand_samp(domain).double()
        y = obj_fn(rand_x, noise).double()
        if verbose:
            print("x: {}, y: {}".format(rand_x.numpy(), y.numpy()[0]))
        lx.append(rand_x)
        ly.append(y)
    tx, ty = torch.cat(lx, dim=0), torch.cat(ly, dim=0)
    return tx, ty


def grid(domain, samp_per_dim=10, noise=0):
    '''Generate test inputs and outputs (grid based, deterministic).
    @param domain, tuple minx maxx
    @param objective, function
    @param samp_per_dim, # samples to take for each dimension (i.e. 10 samp, 6 dim-> 10^6 total samples)
    return tensor of inputs, tensor of outputs
    '''
    xrange = domain[1] - domain[0]
    num_dim = xrange.size(-1)
    l = []
    for x in range(num_dim):
        l.append(torch.linspace(domain[0][x].item(), domain[1][x].item(), steps=samp_per_dim))
    results = torch.meshgrid(l)
    r = []
    for t in results:
        t = torch.reshape(t, (-1, 1))
        r.append(t)
    x = torch.cat(r, -1).float()
    return x


def test_grid(domain: tuple[Tensor, Tensor],
              objective: ObjectiveFunc,
              samp_per_dim: int = 10,
              noise: float = 0.
              ) -> tuple[Tensor, Tensor]:
    '''Generate test inputs and outputs (grid based, deterministic).

    Args
        domain: min_x, max_x
        objective: function
        samp_per_dim: num samples to take for each dimension
            i.e., 10 samp, 6 dim -> 10^6 total samples
        noise: optional noise

    Returns: x, y
    '''
    min_x, max_x = domain
    xrange = max_x - min_x
    num_dim = xrange.size(-1)
    l = []
    for x in range(num_dim):
        l.append(torch.linspace(min_x[0][x].item(), max_x[0][x].item(), steps=samp_per_dim))
    results = torch.meshgrid(l)
    r = []
    for t in results:
        t = torch.reshape(t, (-1, 1))
        r.append(t)
    x = torch.cat(r, -1).float()
    y = []
    y = objective(x)
    return x,y


def samp_discrete_X(num_samp, disc_X):
    inds = set()
    for n in range(num_samp):
        i = random.randrange(0, disc_X.size(0))
        while i in inds:
            i = random.randrange(0, disc_X.size(0))
        inds.add(i)
    # TODO: sorting seems unnecessary...
    sorted_inds = sorted(inds)
    xt = []
    for n in sorted_inds:
        xt.append(torch.reshape(disc_X[n], (1, -1)))
    return torch.cat(xt, 0)


def samp_discrete(n: int, objective: Objective, seed: int, 
                  rng: np.random.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = objective.get_points()
    if rng is None:
        rng = np.random.default_rng(seed)
    inds = np.sort(rng.choice(len(X), size=n))
    return X[inds], y[inds], torch.tensor(inds)

##### for normalizing data points


def norm_domain(domain):
    '''Normalize any domain to be (-1, 1).
    @param domain, a tuple of torch tensors
    return normalized domain, reversion function, conversion function.'''
    xrange = (domain[1] - domain[0])
    # if dim has no range, then make range=1 so doesn't nan/inf
    xrange[xrange==0] = 1
    xrange = xrange.float()
    # scale down to range of 1, no 2
    high = 2*domain[1] / xrange
    low = 2*domain[0] / xrange
    dist_zero = low
    # shift to low at -1
    low = low - dist_zero - 1
    high = high - dist_zero - 1
    # norm -> original
    def norm_revert(inp, dist_zero=dist_zero, xrange=xrange):
        result = inp + dist_zero + 1
        result = result * xrange / 2
        return result
    # original -> norm
    def norm_convert(inp, dist_zero=dist_zero, xrange=xrange):
        result = 2*inp / xrange
        result = result - dist_zero - 1
        return result
    return (low, high), norm_revert, norm_convert


def convert_inputs(inputs, func):
    '''Takes in tensor of concat'd inputs, and calls function on each.
    Returns a new tensor of same dim.'''
    temp = torch.empty(0).float()
    for n in np.arange(inputs.size(0)):
        empt = torch.empty(1, inputs[0].size(0)).float()
        temp = torch.cat((temp, empt), 0)
    for x in np.arange(inputs.size()[0]):
        input = inputs[x]
        conv = func(input)
        temp[x] = conv.float()
    return temp


def calc_mae(inputs, outputs, model, float=False):
    m = model.model
    m.eval()
    with torch.no_grad():
        if gpu or torch.cuda.is_available():
            inputs = inputs.cuda()
            outputs = outputs.cuda()
            m = m.cuda()
        try:
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False):
                if float:
                    preds = m(inputs.float())
                    mae = torch.mean(torch.abs(preds.mean - outputs.float()))
                else:
                    preds = m(inputs)
                    mae = torch.mean(torch.abs(preds.mean - outputs))
            if gpu or torch.cuda.is_available():
                inputs = inputs.cpu()
                outputs = outputs.cpu()
                # model = model.cpu()
                torch.cuda.empty_cache()
        except:
            print("mae error")
            return torch.tensor(0.0).float()
        return mae.cpu().float()


def domain_discrete(x: Tensor) -> tuple[Tensor, Tensor]:
    '''Computes the lower and upper limits of the domain.

    Args:
        x: shape [n, d], all possible discrete inputs

    Returns: (lower, upper), where each is a Tensor of shape [d]
    '''
    lower = x.min(dim=0)[0]
    upper = x.max(dim=0)[0]
    return lower, upper


def query_discrete(X: Tensor, y: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args
        X: shape [n, d]
        y: shape [n]
        x: shape [d]

    Returns: (x', y') where x' is the closest entry in X to x by L1 norm, and
        y' is the corresponding y-value.
    """
    dists = torch.linalg.norm(X - x, ord=1, dim=1)
    closest = dists.argmin()
    return X[closest], y[closest]



def get_closest_discrete(X, num, x):
    """Returns closest n points."""
    dict = {}
    next = 0
    for n in range(X.size(0)):
        temp = torch.mean(torch.abs(x - X[n])).item()
        if len(dict.keys()) < num:
            dict[temp] = torch.reshape(X[n], (1, -1))
            if temp > next:
                next = temp
        elif temp < next:
            del dict[next]
            dict[temp] = torch.reshape(X[n], (1, -1))
            next = np.max(list(dict.keys()))
    try:
        res = torch.cat(list(dict.values()), 0)
    except:
        for val in dict.values():
            print(val.size())
        exit(4)
    return res


def get_freer_gpu():
    gpu = 0
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu = np.argmax(memory_available)
    except:
        gpu = 0
    return gpu


def set_freer_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(get_freer_gpu())
    print('Using GPU {}'.format(str(get_freer_gpu())))


########## optimization for acq--kinda deprecated currently
def torch_optimize(max_obj, batch, bounds, lr, num_iter, epsilon):
    batch.requires_grad = True
    acqoptimizer = torch.optim.Adam([{'params':batch, 'constraints':torch.distributions.constraints.interval(-1, 1)}], lr=lr)
    prev = 0
    # diff = math.inf
    iter = 0
    while iter < num_iter:
        # Zero backprop gradients
        acqoptimizer.zero_grad()
        # Calc loss and backprop derivatives
        output = -max_obj(batch)
        # lossfn is just acq value (want to maximize, so make negative)
        acqloss = output.sum()
        acqloss.backward()
        acqoptimizer.step()
        iter += 1
        if abs(prev - acqloss.item()) < epsilon:
            print('Iter %d - Loss: %.3f (converged)' % (iter, acqloss.item()))
            break
        prev = acqloss.item()
        # if ((iter)%5 == 0 or iter == 1):
        if iter == 1:
            print('Iter %d - Loss: %.3f' % (iter, prev))
        # make sure values are staying in domain
        with torch.no_grad():
            # okay assuming will norm all dim to -1, 1
            batch = torch.clamp(batch, min=-1, max=1)
    del acqloss, output
    batch.requires_grad = False
    return batch


def scipy_optimize(min_obj, batch, bounds):
    coll = []
    min_val, min_x = -math.inf, None
    f_bounds = []
    for low, high in zip(bounds[0][0].detach().numpy(), bounds[1][0].detach().numpy()):
        f_bounds.append((low, high))
    for n in range(batch.size(0)):
        x = batch[n].detach()
        res = scipy.optimize.minimize(min_obj, x0=x, bounds=f_bounds, method='L-BFGS-B')
        # min_val = res.fun[0]
        # min_x = res.x
        min_x = torch.reshape(torch.from_numpy(res.x), (1, -1))
        coll.append(min_x)
        print(n)
    return torch.cat(coll, 0).float()


def find_x(x, queries_x):
    """given continuous optimization, find "closest" discrete data point."""
    # TODO: is this behavior ever used? not particularly sensible.
    if x is None or queries_x is None:
        return True
    for i in range(queries_x.size(0)):
        if torch.equal(torch.reshape(x, (1,-1)), torch.reshape(queries_x[i], (1,-1))):
            return True
    return False
