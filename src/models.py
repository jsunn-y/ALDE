from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Literal
import warnings

import gpytorch
import src.networks as networks
from src.networks import NET_ARGS, OPT_ARGS, SAMP_ARGS
import torch
from torch import Tensor

gpu = torch.cuda.is_available()


class Model:
    """Generic class for models, including GP and deep kernel models.
    common: init (with training, and other), train, evaluate"""

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
        """Initializes a Model object (e.g. GP, deep kernel) and trains it.
        The surrogate model can be extracted by calling .model on the init'd
        Model object.

        Args
            train_x: training inputs
            train_y: training outputs
            min_noise: optional double, minimum-noise GP constraint
            num_iter: number of training iterations
            path: path to save model state_dict
            mtype: one of ['DKL', 'GP']
            kernel: one of ['RBF', 'Lin']
            architecture: for DNN (only DK), list of hidden layer sizes
            dropout: TODO
            mcdropout: TODO
            lr: learning rate
            verbose: int btwn 0, 3 inclusive
        """
        self.mtype = mtype
        self.dkl = "DKL" in mtype.upper() or "CDKL" in mtype.upper()

        # self.path = path
        self.device = "cuda" if gpu else "cpu"

        # setup model structs, likelihood
        inference_args = OPT_ARGS(lr, num_iter)
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
            inference_args,
        )
        if 'BOTORCH' in mtype:
            self.model = networks.BoTorchGP(**self.model_args)
        else:
            self.model = networks.GP(**self.model_args)

    def train(
        self, train_x, train_y, iter=0, track_lc=False, reset=True, dynamic_arc=None
    ):
        if reset:
            self.model_args.train_x, self.model_args.train_y = train_x, train_y
            self.model = networks.GP(**self.model_args)
            train_x, train_y = train_x.to(self.device).double(), train_y.to(self.device).double()
            self.model.likelihood, self.model = self.model.likelihood.to(self.device).double(), self.model.to(self.device).double()
        
        self.model.train_model(train_x, train_y, **self.model_args.inference_args)
        

        return None

