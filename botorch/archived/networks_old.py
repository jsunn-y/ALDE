import torch
import gpytorch
# import torch_geometric
# import torch_geometric.nn as geom_nn
# import torch_geometric.data as geom_data

import utils#, nts


# TODO: standard BNN with torchbnn or blitz?

# TODO: NTS wrapper
class NTS:
    def __init__(self, architecture):
        self.model = nts.NeuralTSDiag(dim, lamdba=1, nu=1, hidden=100, style='ts')


# standard DNN, feedforward
class DNN_FF(torch.nn.Sequential):
    def __init__(self, architecture, activation, dropout, inp_dim=-1):
        super(DNN_FF, self).__init__()

        act_dict = {
        'relu' : torch.nn.ReLU(),
        'lrelu' : torch.nn.LeakyReLU(),
        'swish' : torch.nn.SiLU(),
        'sigmoid' : torch.nn.Sigmoid(),
        'tanh' : torch.nn.Tanh(),
        'softmax' : torch.nn.Softmax(),
        }
        if architecture is None:
            #architecture = [inp_dim, 1000, 500, 50, 5] #for TAPE
            architecture = [inp_dim, 500, 250, 25, 5] #for ESM1b
            #architecture = [inp_dim, 40, 20, 10, 5] #for onehot

        for dim in range(len(architecture)):
            name = str(dim + 1)
            if dim + 1 < len(architecture):
                self.add_module('linear' + name, torch.nn.Linear(architecture[dim], architecture[dim + 1]))
                # don't dropout from output layer ie add below
            if dim + 2 < len(architecture):
                if dropout > 0 and dropout < 1:
                    self.add_module('dropout' + name, torch.nn.Dropout(p=dropout))
                name = activation + name
                try:
                    layer = act_dict[activation.lower()]
                except:
                    layer = act_dict['relu']
                self.add_module(name, layer)

# graph DNN
# TODO: install pyg and pytorch same versions

# gnn_layer_by_name = {
#     "GCN": geom_nn.GCNConv,
#     "GAT": geom_nn.GATConv,
#     "GraphConv": geom_nn.GraphConv
# }

# # graph level
# class GNN_Graph(nn.Module):
    
#     def __init__(self, architecture, dp_rate_linear=0.5, dp_rate_gnn=0.1, **kwargs):
#         """
#         Inputs:
#             c_in - Dimension of input features
#             c_hidden - Dimension of hidden features
#             c_out - Dimension of output features (usually number of classes)
#             dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
#             kwargs - Additional arguments for the GNNModel object
#         """
#         super().__init__()
#         self.GNN = GNN_Node(architecture=architecture[:-1], # Not our prediction output yet!
#                             dp_rate=dp_rate_gnn,
#                             **kwargs)
#         self.head = nn.Sequential(
#             nn.Dropout(dp_rate_linear),
#             nn.Linear(architecture[-2], architecture[-1])
#         )

#     def forward(self, x, edge_index, batch_idx):
#         """
#         Inputs:
#             x - Input features per node
#             edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
#             batch_idx - Index of batch element for each node
#         """
#         x = self.GNN(x, edge_index)
#         x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
#         x = self.head(x)
#         return x

# # node GNN
# class GNN_Node(nn.Module):
    
#     def __init__(self, architecture, layer_name="GCN", dp_rate=0.1, **kwargs):
#         """
#         Inputs:
#             c_in - Dimension of input features
#             c_hidden - Dimension of hidden features
#             c_out - Dimension of the output features. Usually number of classes in classification
#             num_layers - Number of "hidden" graph layers
#             layer_name - String of the graph layer to use
#             dp_rate - Dropout rate to apply throughout the network
#             kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
#         """
#         super().__init__()
#         gnn_layer = gnn_layer_by_name[layer_name]
        
#         layers = []
#         for l_idx in range(len(architecture)-1):
#             layers += [
#                 gnn_layer(in_channels=architecture[l_idx], 
#                           out_channels=architecture[l_idx+1],
#                           **kwargs),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dp_rate)
#             ]
#         self.layers = nn.ModuleList(layers)
    
#     def forward(self, x, edge_index):
#         """
#         Inputs:
#             x - Input features per node
#             edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
#         """
#         for l in self.layers:
#             # For graph layers, we need to add the "edge_index" tensor as additional input
#             # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
#             # we can simply check the class type.
#             if isinstance(l, geom_nn.MessagePassing):
#                 x = l(x, edge_index)
#             else:
#                 x = l(x)
#         return x


# standard GP model

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, outdim, base_kernel, grid_size, aux, dkl=None):
        '''Init GP.
            -@param: training inputs (torch.tensor)
            -@param: training outputs (torch.tensor) corr. to inputs
            -@param: likelihood func(usually mll)
            -@param: outdim, depth of last layer of DNN
            -@param: a kernel (e.g. RBF, grid interp, spectral mixture, etc)
            -@param: grid_size, size of grid for grid interpolation
            -@param: aux variable (used for smoothness constant, etc.)
        '''
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.dkl = (dkl != None)
        self.feature_extractor = dkl
        self.lin = False

        self.num_outputs = 1 # make this input if we do composite fns?
        self._has_transformed_inputs = True
        self._original_train_inputs = None # can fill this later

        self.mean_module = gpytorch.means.ConstantMean()
        if not self.dkl:
            # TODO: redundant
            inpdim = train_x.size(-1)
            outdim = inpdim

        if base_kernel == None or base_kernel.lower() == 'rbf':
            base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(has_lengthscale=True, ard_num_dims=outdim, num_dims=outdim))
            self.covar_module = base_kernel
        elif base_kernel.lower() in ['lin', 'linear']:
            self.covar_module = gpytorch.kernels.LinearKernel()
            self.lin = True
        elif base_kernel.lower() in ['matern', 'mat']:
            # default smoothness param
            nu = 1.5
            if aux in [.5, 1.5, 2.5]:
                nu = aux
            base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu, has_lengthscale=True, ard_num_dims=outdim, num_dims=outdim))
            self.covar_module = base_kernel
        elif base_kernel.lower() == 'sm':
            # example n_mixtures is 4
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=aux, ard_num_dims=outdim, num_dims=outdim)
        else:
            raise ValueError('Base kernel not defined.')
        # can change input kernel, but GridInterpolationKernel will be fixed (SKI)
        # num dims needs to be outdim of DNN (i.e. last layer depth)
        # self.covar_module = base_kernel
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(base_kernel, num_dims=outdim, grid_size=grid_size)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        if self.dkl:
            nnx = self.feature_extractor(x)
        else:
            nnx = x
        mean_x = self.mean_module(nnx)
        covar_x = self.covar_module(nnx)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x):
        # for use with TS acq
        if self.dkl:
            return self.feature_extractor(x)

    def posterior(self, X=None, posterior_transform=None):
        # to conform to botorch model class
        self.eval()
        return self.forward(X)