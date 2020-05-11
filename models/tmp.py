import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNModelVAE(nn.Module):
    def __init__(self, data, hidden_dim1=32, hidden_dim2=16, dropout=0.):
        super(GCNModelVAE, self).__init__()
        input_feat_dim = data.num_features
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return data.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=data.weight_mat)

    def loss_function(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        print(recon_loss, kl)
        return recon_loss + kl

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data):
        x, adj = data.features, data.adj
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return {'adj_recon': self.dc(z), 'mu': mu, 'logvar': logvar}


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
