from UFGConv import Net
from ChebnetII_pro import HypergraphGuidedChebNetProp
import torch.nn.functional as F

import torch
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.linear1 = nn.Linear(hid_dim, hid_dim * 2)
        self.linear1_res = nn.Linear(hid_dim, hid_dim * 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim * 2, hid_dim)
        self.linear2_res = nn.Linear(hid_dim * 2, hid_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hid_dim, n_classes)
    def forward(self, x):
        out1 = self.linear1(x)
        res1 = self.linear1_res(x)
        out1 = self.relu1(out1 + res1)
        out2 = self.linear2(out1)
        res2 = self.linear2_res(out1)
        out2 = self.relu2(out2 + res2)
        out3 = self.linear3(out2)
        return out3

class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def sample_negative_pairs(self, h1, h2, h3, h4):
        num_nodes = h1.size(0)
        neg_idx = torch.randperm(num_nodes)[:int(num_nodes * 0.7)]
        return torch.cat([h3[neg_idx], h4[neg_idx]], dim=0)

    def compute_positive_similarity(self, h1, h2, c):
        h1_norm = F.normalize(h1, dim=-1)
        h2_norm = F.normalize(h2, dim=-1)
        c_norm = F.normalize(c, dim=-1)
        pos_sim1 = torch.sum(h1_norm * c_norm, dim=-1)
        pos_sim2 = torch.sum(h2_norm * c_norm, dim=-1)
        return torch.cat([pos_sim1, pos_sim2], dim=0)

    def compute_negative_similarity(self, h1, h2, neg_samples):
        h1_norm = F.normalize(h1, dim=-1)
        h2_norm = F.normalize(h2, dim=-1)
        neg_norm = F.normalize(neg_samples, dim=-1)
        neg_sim1 = torch.mm(h1_norm, neg_norm.t())
        neg_sim2 = torch.mm(h2_norm, neg_norm.t())
        return torch.cat([neg_sim1, neg_sim2], dim=0)

    def adaptive_temperature(self, pos_sim, neg_sim, init_temp=0.1):
        pos_mean = pos_sim.mean()
        neg_mean = neg_sim.mean()
        gap = torch.abs(pos_mean - neg_mean)
        return init_temp * torch.clamp(gap, min=0.5, max=2.0)

    def forward(self, h1, h2, h3, h4, c, temperature=0.1):
        c_x = c.expand_as(h1).contiguous()

        sc_1 = self.bilinear(h2, c_x).squeeze(1)
        sc_2 = self.bilinear(h1, c_x).squeeze(1)
        sc_3 = self.bilinear(h4, c_x).squeeze(1)
        sc_4 = self.bilinear(h3, c_x).squeeze(1)
        logits = torch.cat((sc_1, sc_2, sc_3, sc_4))

        neg_samples = self.sample_negative_pairs(h1, h2, h3, h4)
        pos_sim = self.compute_positive_similarity(h1, h2, c)
        neg_sim = self.compute_negative_similarity(h1, h2, neg_samples)
        temp = self.adaptive_temperature(pos_sim, neg_sim, temperature)

        pos_exp = torch.exp(pos_sim / temp)
        neg_exp_sum = torch.sum(torch.exp(neg_sim / temp), dim=1)
        neg_exp_sum = torch.clamp(neg_exp_sum, min=1e-10, max=1e10)
        contrast_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum)).mean()

        return logits, contrast_loss



class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, K, dprate, dropout, is_bns, act_fn, r, Lev, num_nodes, device,
                 highpass_filter, lowpass_filter):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.dprate = dprate
        self.dropout = dropout
        self.is_bns = is_bns
        self.act_fn = nn.ReLU() if act_fn == 'relu' else nn.ELU()
        self.r = r
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.device = device
        self.encoder_high = HypergraphGuidedChebNetDOS(
            num_features=in_dim,
            hidden=hidden_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            filter_type=highpass_filter,
            num_nodes=num_nodes
        )
        self.encoder_low = HypergraphGuidedChebNetDOS(
            num_features=in_dim,
            hidden=hidden_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn,
            filter_type=lowpass_filter,
            num_nodes=num_nodes
        )
        self.encoder2 = Net(
            num_features=in_dim,
            nhid=hidden_dim,
            r=r,
            Lev=Lev,
            num_nodes=num_nodes,
            shrinkage='soft',
            threshold=1e-4,
            dropout_prob=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)
        self.proj_h1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_h5 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_fused = nn.Linear(2 * hidden_dim, hidden_dim)
        self.disc = Discriminator(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder_high.reset_parameters()
        self.encoder_low.reset_parameters()
        self.encoder2.reset_parameters()
        nn.init.xavier_uniform_(self.proj_h1.weight)
        nn.init.xavier_uniform_(self.proj_h2.weight)
        nn.init.xavier_uniform_(self.proj_h5.weight)
        nn.init.xavier_uniform_(self.proj_fused.weight)
        self.proj_h1.bias.data.fill_(0)
        self.proj_h2.bias.data.fill_(0)
        self.proj_h5.bias.data.fill_(0)
        self.proj_fused.bias.data.fill_(0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm5.reset_parameters()

    def get_components(self, x, hyperedge_index, edge_weight, d_list):
        h1 = self.encoder_high(x=x, hyperedge_index=hyperedge_index, edge_weight=edge_weight)
        h2 = self.encoder_low(x=x, hyperedge_index=hyperedge_index, edge_weight=edge_weight)
        h5 = self.encoder2(x=x, d_list=d_list if d_list is not None else None)
        h1 = self.norm1(h1)
        h2 = self.norm2(h2)
        h5 = self.norm5(h5)
        h1 = self.act_fn(h1)
        h2 = self.act_fn(h2)
        h5 = self.act_fn(h5)
        h1 = self.proj_h1(h1)
        h2 = self.proj_h2(h2)
        h5 = self.proj_h5(h5)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h5 = F.dropout(h5, p=self.dropout, training=self.training)
        return h1, h2, h5

    def forward(self, hyperedge_index, feat, shuf_feat, edge_weight, d_list):

        feat = feat.float().to(self.device)
        shuf_feat = shuf_feat.float().to(self.device)
        hyperedge_index = hyperedge_index.to(self.device)
        edge_weight = edge_weight.float().to(self.device) if edge_weight is not None else None
        feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)
        shuf_feat = (shuf_feat - shuf_feat.mean(dim=0)) / (shuf_feat.std(dim=0) + 1e-6)
        h1, h2, h5 = self.get_components(feat, hyperedge_index, edge_weight, d_list)
        h3, h4, h5_neg = self.get_components(shuf_feat, hyperedge_index, edge_weight, d_list)
        h_cheb = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        h_cheb = self.act_fn(h_cheb)
        h_cheb = F.normalize(h_cheb, p=2, dim=-1)
        h5 = F.normalize(h5, p=2, dim=-1)
        fused = torch.cat([h_cheb, h5], dim=-1)
        fused_out = self.proj_fused(fused)
        logits, contrasloss = self.disc(h1, h2, h3, h4, fused_out)
        return logits, contrasloss

    def get_embedding(self, hyperedge_index, feat, edge_weight, d_list):
        feat = feat.float().to(self.device)
        hyperedge_index = hyperedge_index.to(self.device)
        edge_weight = edge_weight.float().to(self.device) if edge_weight is not None else None
        feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)
        h1, h2, h5 = self.get_components(feat, hyperedge_index, edge_weight, d_list)
        h_cheb = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)
        h_cheb = self.act_fn(h_cheb)
        h_cheb = F.normalize(h_cheb, p=2, dim=-1)
        h5 = F.normalize(h5, p=2, dim=-1)
        fused = torch.cat([h_cheb, h5], dim=-1)
        fused_out = self.proj_fused(fused)
        return fused_out

class HypergraphGuidedChebNet(nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False,
                 act_fn='relu', filter_type='g_high_pass', num_nodes=None):
        super(HypergraphGuidedChebNet, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.prop1 = HypergraphGuidedChebNetProp(
            in_channels=num_features,
            out_channels=num_features,
            K=K,
            num_nodes=num_nodes,
            filter_type=filter_type
        )
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        self.bn.reset_parameters()

    def forward(self, x, hyperedge_index, edge_weight=None):
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, hyperedge_index, edge_weight=edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.is_bns:
            x = self.bn(x)
        x = self.act_fn(x)
        return x


class HypergraphGuidedChebNetDOS(nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False,
                 act_fn='relu', filter_type='DOS', num_nodes=None):
        super(HypergraphGuidedChebNetDOS, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.prop1 = HypergraphGuidedChebNetProp(
            in_channels=num_features,
            out_channels=num_features,
            K=K,
            num_nodes=num_nodes,
            filter_type=filter_type
        )
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        self.bn.reset_parameters()

    def forward(self, x, hyperedge_index, edge_weight=None):
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, hyperedge_index, edge_weight=edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.is_bns:
            x = self.bn(x)
        x = self.act_fn(x)
        return x
