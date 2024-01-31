from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import gpytorch
import src.networks as networks
from src.networks import NET_ARGS
import torch
from torch import Tensor

class Model:
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        min_noise: float | None,
        num_iter: int,
        path: str,
        mtype: Literal["DKL", "CDKL", "GP"] = "DKL",
        kernel: Literal["RBF", "Lin"] | None = None,
        architecture: Sequence[int] | None = None,
        activation: str | None = None,
        dropout: float = 0.0,
        mcdropout: float = 0.0,
        lr: float = 0.01,
        verbose: int = 1,
    ):
        """Initializes a Model object for training (e.g. GP, DNN_ENSEMBLE, or DKL) and trains it. Note that BOOSTING_ENSEMBLE does not use this model.

        Args
            train_x: training inputs
            train_y: training outputs
            min_noise: optional double, minimum-noise GP constraint
            num_iter: number of training iterations
            path: path to save model state_dict
            mtype: one of ['GP_BOTORCH', 'DKL_BOTORCH', 'DNN_ENSEMBLE']
            kernel: one of ['RBF']
            architecture: list of hidden layer sizes
            dropout: training dropout for neural network
            mcdropout: test time dropout for neural network
            lr: learning rate
            verbose: int btwn 0, 3 inclusive
        """
        self.mtype = mtype
        self.dkl = "DKL" in mtype.upper()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        noise_constraint = (
            gpytorch.constraints.GreaterThan(min_noise)
            if min_noise != None
            else None
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )

        self.model_args = NET_ARGS(
            train_x,
            train_y,
            architecture,
            activation,
            kernel,
            dropout,
            likelihood,
            self.device,
        )

    def train(
        self, train_x, train_y, iter=0, track_lc=False, reset=True, dynamic_arc=None
    ):
        if reset:
            self.model_args.train_x, self.model_args.train_y = train_x, train_y
            
            if 'DNN' in self.mtype and 'ENSEMBLE' in self.mtype:
                self.model = networks.DNN_FF(**self.model_args)
            else:
                if 'BOTORCH' in self.mtype:
                    self.model = networks.BoTorchGP(**self.model_args)
                else:
                    self.model = networks.GP(**self.model_args)
                self.model.likelihood = self.model.likelihood.to(self.device).double()

            train_x, train_y = train_x.to(self.device).double(), train_y.to(self.device).double()
            self.model =  self.model.to(self.device).double()
        
        self.model.train_model(train_x, train_y)

        return None

