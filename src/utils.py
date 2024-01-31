from __future__ import annotations

from collections.abc import Callable, Sequence, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

gpu = torch.cuda.is_available()

Noise = Tensor 
ObjectiveFunc = Callable[[Tensor], Tensor]

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

def samp_discrete(n: int, objective: Objective, seed: int, 
                  rng: np.random.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = objective.get_points()
    if rng is None:
        rng = np.random.default_rng(seed)
    inds = np.sort(rng.choice(len(X), size=n))
    return X[inds], y[inds], torch.tensor(inds)


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

def find_x(x, queries_x):
    """given continuous optimization, find "closest" discrete data point."""
    # TODO: is this behavior ever used? not particularly sensible.
    if x is None or queries_x is None:
        return True
    for i in range(queries_x.size(0)):
        if torch.equal(torch.reshape(x, (1,-1)), torch.reshape(queries_x[i], (1,-1))):
            return True
    return False
