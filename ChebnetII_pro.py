import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import numpy as np
import scipy.sparse as ss


def add_self_loops(edge_index, edge_weight, fill_value, num_nodes):

    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    loop_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    return edge_index, edge_weight


def get_hypergraph_laplacian(hyperedge_index, n_nodes, edge_weight=None, normalized=True):

    hyperedge_index = hyperedge_index.cpu().numpy()
    n_edges = hyperedge_index[1].max() + 1 if hyperedge_index.size > 0 else 0

    if n_edges == 0:
        H = ss.csr_matrix((n_nodes, 0))
    else:
        data = np.ones(hyperedge_index.shape[1])
        H = ss.coo_matrix(
            (data, (hyperedge_index[0], hyperedge_index[1])),
            shape=(n_nodes, n_edges)
        ).tocsr()

    if edge_weight is None:
        edge_weight = np.ones(n_edges, dtype=np.float64)
    else:
        edge_weight = edge_weight.cpu().numpy()


    Dv = np.array(H.sum(axis=1)).squeeze()
    De = np.array(H.sum(axis=0)).squeeze()


    if normalized:
        Dv_inv_sqrt = np.where(Dv > 0, 1.0 / np.sqrt(Dv), 0)
        De_inv = np.where(De > 0, 1.0 / De, 0)
        W = ss.diags(edge_weight)
        L = ss.eye(n_nodes) - ss.diags(Dv_inv_sqrt) @ H @ W @ ss.diags(De_inv) @ H.T @ ss.diags(Dv_inv_sqrt)
    else:
        W = ss.diags(edge_weight)
        L = ss.diags(Dv) - H @ W @ H.T

    L_coo = L.tocoo()
    edge_index = torch.tensor([L_coo.row, L_coo.col], dtype=torch.long)
    norm = torch.tensor(L_coo.data, dtype=torch.float32)

    return edge_index, norm



def moments_cheb_dos_hypergraph(hyperedge_index, n_nodes, edge_weight=None, nZ=30, N=30, device='cpu'):
    hyperedge_index = hyperedge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    edge_index, norm = get_hypergraph_laplacian(hyperedge_index, n_nodes, edge_weight)
    edge_index = edge_index.to(device)
    norm = norm.to(device)
    class LaplacianProp(MessagePassing):
        def __init__(self):
            super(LaplacianProp, self).__init__(aggr='add')
        def forward(self, x, edge_index, norm):
            return self.propagate(edge_index, x=x, norm=norm)
        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j
    prop = LaplacianProp().to(device)
    def Afun(x):
        return prop(x, edge_index, norm)
    Z = np.random.choice([-1, 1], size=(n_nodes, nZ))
    Z = torch.from_numpy(Z).float().to(device)
    cZ = moments_cheb(Afun, Z, N)
    c = np.mean(cZ.cpu().numpy(), axis=1).reshape(N, -1)
    cs = np.std(cZ.cpu().numpy(), axis=1, ddof=1).reshape(N, -1) / np.sqrt(nZ)
    return c, cs
def moments_cheb(Afun, V, N=10):
    n, p = V.shape
    c = torch.zeros((N, p), device=V.device)
    TVp = V
    TVk = Afun(TVp)
    c[0] = torch.sum(V * TVp, dim=0)
    c[1] = torch.sum(V * TVk, dim=0)
    for i in range(2, N):
        TV = 2 * Afun(TVk) - TVp
        TVp = TVk
        TVk = TV
        c[i] = torch.sum(V * TVk, dim=0)
    return c

def t_polynomial_zeros_hypergraph(x0=0.0, x1=2.0, n=10):
    return (x1 - x0) * (np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi) + 1) / 2 + x0


def g_high_pass_hypergraph(x):

    return 1 - np.exp(-25 * x ** 2)


def g_low_pass_hypergraph(x):

    return np.exp(-38 * x ** 2)


def filter_jackson(c):

    N = len(c)
    n = np.arange(N)
    tau = np.pi / (N + 1)
    g = ((N - n + 1) * np.cos(tau * n) + np.sin(tau * n) / np.tan(tau)) / (N + 1)
    return c * g


def polyfit_chebyshev(x, y, n, x0, x1):

    omega = (x1 - x0) / 2
    rho = -((x1 + x0) / (x1 - x0))
    IN = np.identity(n + 1)
    X = np.diag(x)
    firstElement = (2 / omega) * X + 2 * rho * IN
    firstRow = np.concatenate((firstElement, -IN), axis=1)
    secondRow = np.concatenate((IN, np.zeros((n + 1, n + 1))), axis=1)
    Xcurly = np.concatenate((firstRow, secondRow), axis=0)

    m = x.size
    T = np.ones((m, 1))
    x = x.reshape(-1, 1)
    Q = np.concatenate((x, T), axis=0)
    H = np.zeros((n + 1, n))

    for k in range(n):
        q = np.matmul(Xcurly, Q[:, k])
        for j in range(k):
            H[j, k] = np.dot(Q[:, j].T, (q / m))
            q = q - np.dot(H[j, k], Q[:, j])
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(m)
        Q = np.column_stack((Q, q / H[k + 1, k]))

    newQ = Q[n + 1:2 * (n + 1), :]
    d = np.linalg.solve(newQ.astype(np.float64), y.astype(np.float64))
    return d, H


def compare_fit_panelA_hypergraph(f, degree, x0=0.0, x1=2.0):

    x = t_polynomial_zeros_hypergraph(x0, x1, degree)
    y = f(x)
    coefficients, _ = polyfit_chebyshev(x, y, degree - 1, x0, x1)
    return coefficients


class HypergraphGuidedChebNetProp(MessagePassing):
    def __init__(self, in_channels, out_channels, K, num_nodes, filter_type='g_high_pass'):
        super(HypergraphGuidedChebNetProp, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.num_nodes = num_nodes
        self.filter_type = filter_type
        self.temp = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.lower = 0.0
        self.upper = 2.0
        self.rw_filters = filter_type in ['DOS', 'Linear', 'Eigen']
        self.reset_parameters()
    def reset_parameters(self):
        self.temp.data = torch.from_numpy(t_polynomial_zeros_hypergraph(self.lower, self.upper, self.K)).float()

    def forward(self, x, hyperedge_index, edge_weight=None):

        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        hyperedge_index = hyperedge_index.long().to(x.device)
        if edge_weight is not None:
            edge_weight = torch.where(torch.isnan(edge_weight), torch.zeros_like(edge_weight), edge_weight).to(x.device)

        edge_index, norm = get_hypergraph_laplacian(hyperedge_index, self.num_nodes, edge_weight)
        edge_index = edge_index.to(x.device)
        norm = norm.to(x.device)
        norm = torch.where(torch.isnan(norm), torch.zeros_like(norm), norm)
        # norm = torch.clamp(norm, min=-2.0, max=2.0)
        if self.filter_type == 'g_high_pass':
            coe_tmp = compare_fit_panelA_hypergraph(g_high_pass_hypergraph, self.K, 0.0, 2.0)
        elif self.filter_type == 'g_low_pass':
            coe_tmp = compare_fit_panelA_hypergraph(g_low_pass_hypergraph, self.K, 0.0, 2.0)
        elif self.filter_type in ['DOS', 'Linear', 'Eigen']:
            c, _ = moments_cheb_dos_hypergraph(hyperedge_index, self.num_nodes, edge_weight, nZ=30, N=30)
            random_elements = np.linspace(0, 2, 100)
            f_lambda = random_elements
            idx = np.round(np.linspace(0, len(f_lambda) - 1, self.K)).astype(int)
            f_lambda = f_lambda[idx]
            x_cheb = t_polynomial_zeros_hypergraph(0.0, 2.0, self.K)
            coe_tmp, _ = polyfit_chebyshev(x_cheb, f_lambda, self.K - 1, 0.0, 2.0)

        coeffs = filter_jackson(coe_tmp)
        coe = torch.from_numpy(coeffs).float().to(x.device)

        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        Tx_0 = x
        if self.rw_filters:
            edge_index_tilde, norm_tilde = add_self_loops(edge_index, norm, fill_value=-1.0, num_nodes=self.num_nodes)
            edge_index_tilde = edge_index_tilde.to(x.device)
            norm_tilde = norm_tilde.to(x.device)
            Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde)
        else:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)

        Tx_1 = torch.where(torch.isnan(Tx_1), torch.zeros_like(Tx_1), torch.clamp(Tx_1, min=-20.0, max=20.0))
        out = coe[0] * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K):
            if self.rw_filters:
                Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde)
            else:
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = torch.where(torch.isnan(Tx_2), torch.zeros_like(Tx_2), torch.clamp(Tx_2, min=-20.0, max=20.0))
            Tx_2 = 2 * Tx_2 - Tx_0
            out += coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = torch.clamp(out, min=-20.0, max=20.0)
        out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out.float()

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, filter_type={self.filter_type})'
