from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Literal
import warnings

import gpytorch
import networks
from networks import NET_ARGS, OPT_ARGS, SAMP_ARGS
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
        mtype: Literal["BDKL", "DKL", "CDKL", "GP"] = "DKL",
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
        self.bnn = "BDKL" in mtype.upper()

        # self.path = path
        self.device = "cuda" if gpu else "cpu"

        # setup model structs, likelihood
        if self.bnn: # TODO: should be if mcmc in mtype
            inference_args = SAMP_ARGS(100, 100, 1, 2, verbose)
            likelihood = None
        else:
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

        if self.bnn and self.dkl:
            self.model = networks.BDKL_MCMC_numpyro(**self.model_args)
        elif self.bnn:
            raise NotImplementedError()
        elif 'BOTORCH' in mtype:
            self.model = networks.BoTorchGP(**self.model_args)
        else:
            self.model = networks.GP(**self.model_args)
            #print(self.model.train_inputs[0][0,:])
            #print(self.model.train_inputs[0].shape)
            #print(self.model.train_targets)

    def train(
        self, train_x, train_y, iter=0, track_lc=False, reset=True, dynamic_arc=None
    ):
        if not self.bnn:
            if reset:
                self.model_args.train_x, self.model_args.train_y = train_x, train_y
                if 'BOTORCH' in self.mtype:
                    self.model = networks.BoTorchGP(**self.model_args)  # remake
                else:
                    self.model = networks.GP(**self.model_args)
                train_x, train_y = train_x.to(self.device).double(), train_y.to(self.device).double()
                self.model.likelihood, self.model = self.model.likelihood.to(self.device).double(), self.model.to(self.device).double()
        else:
            pass
        
        # if self.mtype == 'CDKL':
        #     train_x = torch.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        self.model.train_model(train_x, train_y, **self.model_args.inference_args)
        
        # if self.model.bnn:
        #     pass
        # elif reset:
        #     self.model = self.model.reset(self.architecture, dropout=self.dropout, train=(train_x, train_y))
        # else:
        #     sd_path = self.path + 'state_dict.pt'
        #     # first save weights
        #     torch.save(self.model.model.state_dict(), sd_path)
        #     # init from scratch with weights and larger dataset
        #     self.model = self.model.reset(self.architecture, dropout=self.dropout, path=sd_path, reuse=True, train=(train_x, train_y))
        #     try:
        #         os.remove(sd_path)
        #     except Exception as e:
        #         print(f'Could not find/remove {sd_path}.\n{e}\n')
        # model, ll, losses, maes = self.model.train(self.num_iter, self.lr, train=(train_x, train_y), verbose=self.verbose, track_lc=track_lc)

        # if self.mcdropout > 0:
        #     dropout = torch.nn.Dropout(p=self.mcdropout)
        #     if self.model.dkl:
        #         d = self.model.model.feature_extractor.state_dict()
        #         for key in d.keys():
        #             if '.weight' in key:
        #                 d[key] = dropout(d[key])
        #         # update dict
        #         self.model.model.feature_extractor.load_state_dict(d)

        return None

