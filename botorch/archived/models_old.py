import math
import os
import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import gpytorch
import networks
import torch
from torch import Tensor

gpu = torch.cuda.is_available()


class Model:
    '''Generic class for models, including GP and deep kernel models.
    common: init (with training, and other), train, evaluate'''

    def __init__(self,
                 train_x: Tensor,
                 train_y: Tensor,
                 min_noise: float | None,
                 num_iter: int,
                 path: str,
                 mtype: Literal['DKL', 'GP'] = 'DKL',
                 kernel: Literal['RBF', 'Lin'] | None = None,
                 architecture: Sequence[int] | None = None,
                 activation: str | None = None,
                 dropout: float = 0.,
                 mcdropout: float = 0.,
                 lr: float = .01,
                 verbose: bool = True):
        '''Initializes a Model object (e.g. GP, deep kernel) and trains it.
        The surrogate model can be extracted by calling .model on the init'd
        Model object.

        Args
            train_x: training inputs
            train_y: training outputs
            min_noise: optional float, minimum-noise GP constraint
            num_iter: number of training iterations
            path: path to save model state_dict
            mtype: one of ['DKL', 'GP']
            kernel: one of ['RBF', 'Lin']
            architecture: for DNN (only DK), list of hidden layer sizes
            dropout: TODO
            mcdropout: TODO
            lr: learning rate
            verbose: bool
        '''
        self.model = BaseModel(architecture, train_x, train_y, min_noise, kernel, mtype, activation=activation, dropout=dropout)
        self.min_noise, self.num_iter, self.mtype, self.kernel, self.architecture, self.activation, self.lr, self.verbose, self.path = min_noise, num_iter, mtype, kernel, architecture, activation, lr, verbose, path
        if dropout is None or dropout <= 0:
            self.dropout = 0.
        else:
            self.dropout = dropout
        if mcdropout is None or mcdropout <= 0:
            self.mcdropout = 0.
        else:
            self.mcdropout = mcdropout

    def train(self, train_x, train_y, iter=0, track_lc=False, reset=True, dynamic_arc=None):
        if reset:
            self.model = self.model.reset(self.architecture, dropout=self.dropout, train=(train_x, train_y))
        else:
            sd_path = self.path + 'state_dict.pt'
            # first save weights
            torch.save(self.model.model.state_dict(), sd_path)
            # init from scratch with weights and larger dataset
            self.model = self.model.reset(self.architecture, dropout=self.dropout, path=sd_path, reuse=True, train=(train_x, train_y))
            try:
                os.remove(sd_path)
            except Exception as e:
                print(f'Could not find/remove {sd_path}.\n{e}\n')
        model, ll, losses, maes = self.model.train(self.num_iter, self.lr, train=(train_x, train_y), verbose=self.verbose, track_lc=track_lc)

        if self.mcdropout > 0:
            dropout = torch.nn.Dropout(p=self.mcdropout)
            if self.model.dkl:
                d = self.model.model.feature_extractor.state_dict()
                for key in d.keys():
                    if '.weight' in key:
                        d[key] = dropout(d[key])
                # update dict
                self.model.model.feature_extractor.load_state_dict(d)

        return self.model, ll, losses, maes


class BaseModel:
    '''Base model object. Can represent GP or DKL.'''

    def __init__(self, architecture, train_x, train_y, min_noise, base_kernel, mtype='DKL', grid_size=10, aux=None, activation='ReLU', dropout=0):
        '''Initialize deep kernel.
            -@param: architecture, a list of layer #s, starting with the input
            dimensions and ending with the output dimension
            -@param: train_x, training inputs
            -@param: train_y, training outputs
            -@param: base_kernel, the inner kernel for the GP applied to the
            DNN (outer is set as GridInterpolationKernel, ScaleKernel as per
            KISS-GP framework)
            -@param: aux, variable for kernel params (e.g. smoothness)
        '''
        self.dkl = (mtype.upper() == 'DKL')
        self.dgkl = False # deep graph kernel

        if self.dgkl:
            print('Not yet implemented.')
            feature_extractor = networks.GNN_Graph(architecture, dp_rate_linear=0.5, dp_rate_gnn=dropout)
        elif self.dkl:
            self.feature_extractor = networks.DNN_FF(architecture, activation, dropout, inp_dim=train_x[0].size(-1))
        else:
            self.feature_extractor = None

        self.architecture, self.base_kernel, self.mtype, self.aux, self.activation, self.min_noise = architecture, base_kernel, mtype, aux, activation, min_noise
        self.train_x = train_x
        self.train_y = train_y
        if self.min_noise is not None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(self.min_noise))
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if architecture is None:
            outdim = -1
        else:
            outdim = architecture[-1]
        self.model = networks.GP(train_x, train_y, self.likelihood, outdim, base_kernel, grid_size, aux, dkl=self.feature_extractor)
        self.double = self.model.lin

    def reset(self, dynamic_arc, dropout=0, path=None, reuse=False, train=None):
        if train is not None:
            self.train_x, self.train_y = train
        # if arcs are different can't load in same weights
        diff_arc = False
        if self.dkl and dynamic_arc is not None:
            diff_arc = (self.architecture != dynamic_arc)
            self.architecture = dynamic_arc
        self = BaseModel(self.architecture, self.train_x, self.train_y, self.min_noise, self.base_kernel, mtype=self.mtype, aux=self.aux, activation=self.activation, dropout=dropout)
        if reuse and path is not None and not diff_arc:
            self.model.load_state_dict(torch.load(path))
        return self

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + 'gpmodel.pt')
        try: # will fail for normal GP model
            torch.save(self.model.feature_extractor.state_dict(), filename + 'dnn.pt')
        except Exception as e:
            print(e)
        print(f"Saved base model to {filename}")

    def train(self, num_iter: int, lr, train=None, verbose=True, track_lc=False):
        '''Trains DK model on training data. Uses Adam optimizer and trains DNN
        and GP hyperparameters for num_iter iterations.
        Detached from self so can be used externally by Model object.
            -@param: model, the GP w/ DNN
            -@param: likelihood, probably mll
            -@param: train_x, training inputs
            -@param: train_y, training outputs
            -@param: lr, learning rate
        '''
        warnings.filterwarnings("ignore", message="The input matches the stored training data")
        if train is None:
            train_x, train_y = self.train_x, self.train_y
        else:
            train_x, train_y = train

        tmodel, tlikelihood, ttrain_x, ttrain_y = self.model, self.likelihood, train_x, train_y
        if self.double:
            tmodel, tlikelihood, ttrain_x, ttrain_y = tmodel.double(), tlikelihood.double(), ttrain_x.double(), ttrain_y.double()
        if gpu:
            tmodel, tlikelihood, ttrain_x, ttrain_y = tmodel.cuda(), tlikelihood.cuda(), ttrain_x.cuda(), ttrain_y.cuda()

        # Find optimal model hyperparameters. 1st switch to train mode.
        tmodel.train()
        tlikelihood.train()

        # Use the adam optimizer
        if self.dkl:
            optimizer = torch.optim.Adam([
                {'params': tmodel.feature_extractor.parameters()},
                {'params': tmodel.covar_module.parameters()},
                {'params': tmodel.mean_module.parameters()},
                {'params': tmodel.likelihood.parameters()},
            ], lr=lr) # weight_decay
        else:
            optimizer = torch.optim.Adam([
                {'params': tmodel.parameters()}
            ], lr=lr)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(tlikelihood, tmodel)

        def train_opt(optimizer, mll):
            startTime = datetime.now()
            prev = 0
            iter = 0
            losses, maes = [], []
            while iter < num_iter:
                optimizer.zero_grad()
                loss, lim, count = 0, 0, 0
                preds = tmodel(ttrain_x)
                loss = -mll(preds, ttrain_y)
                while lim < ttrain_y.size(0):
                    temp = -mll(preds[lim:lim + 100], ttrain_y[lim:lim + 100])
                    loss += temp
                    del temp
                    torch.cuda.empty_cache()
                    lim += 100
                    count += 1
                del preds
                # normalize bc means of batches
                loss /= count
                loss.backward()
                prev = float(loss)
                # losses.append(prev)
                if verbose and (iter % 1 == 0):
                    print('Iter %d - Loss: %.3f' % (iter, prev))
                optimizer.step()
                del loss, prev
                torch.cuda.empty_cache()
                iter += 1
                # if track_lc:
                #     mae = utils.calc_mae(ttrain_x, ttrain_y, tmodel)
                #     maes.append(mae.item())
                #     tmodel.train()
            # print("Iterations: {}, Delta: {}, Time: {}\n".format(iter, losses[-1], datetime.now() - startTime))
            ll, lim, count = 0, 0, 0
            preds = tmodel(ttrain_x)
            with torch.no_grad():
                while lim < ttrain_y.size(0):
                    temp = -mll(preds[lim:lim + 100], ttrain_y[lim:lim + 100])
                    ll += temp
                    lim += 100
                    count += 1
            ll = ll.detach()/count
            # print("Iterations: {}, Delta: {}, Time: {}\n".format(iter, ll, datetime.now() - startTime))
            return torch.reshape(ll, (1,1)), losses, maes

        # See dkl_mnist.ipynb for explanation of this flag
        # with gpytorch.settings.use_toeplitz(True):--for kiss gp
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
            ll, losses, maes = train_opt(optimizer, mll)

        model, likelihood = tmodel.cpu(), tlikelihood.cpu()
        del tmodel, tlikelihood, ttrain_x, ttrain_y
        if gpu:
            torch.cuda.empty_cache()

        # eval mode before ending
        model.eval()
        likelihood.eval()
        self.model, self.likelihood = model, likelihood

        return model, ll, losses, maes
