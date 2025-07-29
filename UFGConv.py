import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse

from scipy.sparse.linalg import lobpcg


def get_dlist(edge_index, norm, args, feat, device):
    num_nodes = feat.shape[0]
    L = sparse.coo_matrix((norm.numpy(),
                       (edge_index[0, :].numpy(),
                        edge_index[1, :].numpy())
                       ),
                      shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    # lambda_max = lambda_max[0]

    # extract decomposition/reconstruction Masks
    FrameType = args.FrameType
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')
    Lev = args.Lev  # level of transform
    s = args.s  # dilation scale
    n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
    # J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # 原文 dilation level to start the decomposition
    J = 2
    r = len(DFilters)
    # get matrix operators
    d = get_operator(L, DFilters, n, s, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
    return r, Lev, d_list


@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)
    return torch.sparse_coo_tensor(index, value, A.shape)

def ChebyshevApprox(f, n):
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c

def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)

    a = np.pi / 2  # consider the domain of masks as [0, pi]
    FD1 = sparse.identity(L.shape[0])
    d = dict()

    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d
class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.crop_len = (Lev - 1) * num_nodes

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        orig_dtype = x.dtype

        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = torch.matmul(x, self.weight.to(torch.float32))
            for d in d_list:
                x = torch.sparse.mm(d, x)

            # Shrinkage
            if self.shrinkage is not None:
                if self.shrinkage == 'soft':
                    x = torch.sign(x) * torch.relu(torch.abs(x) - self.threshold)
                elif self.shrinkage == 'hard':
                    x = x * (torch.abs(x) > self.threshold)
            # Spectral filtering
            x = x * self.filter[:x.size(0)].to(x.device)

            for d in d_list[self.Lev - 1:]:
                x = torch.sparse.mm(d, x)

            # Bias
            if self.bias is not None:
                x += self.bias

        return x.to(orig_dtype)
class Net(nn.Module):
    def __init__(self, num_features, nhid, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, dropout_prob=0.5):
        super(Net, self).__init__()
        self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.drop1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(nhid)  # Added layer normalization
        self.act_fn = nn.ReLU()  # Explicit ReLU activation

    def reset_parameters(self):
        # Initialize UFGConv parameters
        if hasattr(self.GConv1, 'reset_parameters'):
            self.GConv1.reset_parameters()
        else:
            for module in self.GConv1.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                    if module.bias is not None:
                        module.bias.data.fill_(0)
        self.norm1.reset_parameters()

    def forward(self, x, d_list):
        x = x.float()
        # Apply UFGConv
        x = self.GConv1(x, d_list)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.drop1(x)

        return x