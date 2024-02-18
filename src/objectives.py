from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import Tensor

import src.utils as utils
from src.utils import Noise
from src.encoding_utils import generate_onehot, generate_all_combos

class Objective:

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.
                  ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args
            x: shape [batch_size, d], input
            noise: amount of noise

        Returns: shape [batch_size]
        """
        raise NotImplementedError

    @staticmethod
    def get_max() -> Tensor:
        """Returns maximum value of objective function."""
        raise NotImplementedError

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        """Returns (low, high) domain of objective function.

        low, high have type doubleTensor.
        """
        raise NotImplementedError

    @staticmethod
    def get_points() -> tuple[Tensor, Tensor]:
        """Returns (x, y) pairs."""
        raise NotImplementedError

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        """Returns (x, y) pairs."""
        raise NotImplementedError

class Combo(Objective):
    """
    Class for active learning simulations on combinatory libraries.
    """

    def __init__(self, protein, encoding):
        fitness_df = pd.read_csv('data/' + protein + '/fitness.csv')
        self.y = torch.tensor(fitness_df['fitness'].values).double()
        self.y = self.y/self.y.max()
            
        self.X = torch.load('data/' + protein + '/' + encoding + '_x.pt')
        
    def objective(self, x: Tensor, noise: Noise = 0.) -> tuple[Tensor, Tensor]:
        qx, qy = utils.query_discrete(self.X, self.y, x)
        return qx.double(), qy.double()

    def get_max(self) -> Tensor:
        return torch.max(self.y).double()

    def get_domain(self) -> tuple[Tensor, Tensor]:
        lower, upper = utils.domain_discrete(self.X)
        return lower.double(), upper.double()

    def get_points(self) -> tuple[Tensor, Tensor]:
        return self.X.double(), self.y.double()

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Combo.get_points()

class Production(Objective):
    """
    Class for proposing new sequences to screen in a production campaign, on a combinatorial design space.
    """

    def __init__(self, df, protein, encoding, obj_col):
        train_combos = df['Combo'].tolist()
        self.nsamples = len(train_combos)
        self.ytrain = df[obj_col].values
        self.Xtrain = generate_onehot(train_combos)
        self.Xtrain = torch.reshape(self.Xtrain, (self.Xtrain.shape[0], -1))

        name = protein
        
        assert encoding == 'onehot' #currently only works for onehot encodings, but can be extended to other encodings
        self.all_combos = list(pd.read_csv('data/' + name + '/all_combos.csv')['Combo'].values)
        self.train_indices = [self.all_combos.index(combo) for combo in train_combos]

        self.X = torch.load('data/' + name + '/onehot_x.pt')

        #filler array,used to measure regret, does not affect outcome
        self.y = np.zeros(len(self.all_combos))

        self.y[self.train_indices] = self.ytrain
        self.ytrain = torch.tensor(self.ytrain)
        self.y = torch.tensor(self.y)
        self.train_indices = torch.tensor(self.train_indices)

    def objective(self, x: Tensor, noise: Noise = 0.) -> tuple[Tensor, Tensor]:
        qx, qy = utils.query_discrete(self.X, self.y, x)
        return qx.double(), qy.double()

    def get_max(self) -> Tensor:
        return torch.max(self.y).double()

    def get_domain(self) -> tuple[Tensor, Tensor]:
        lower, upper = utils.domain_discrete(self.X)
        return lower.double(), upper.double()

    def get_points(self) -> tuple[Tensor, Tensor]:
        return self.X.double(), self.y.double()

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Production.get_points()
    
ALL_OBJS = [Combo, Production]
