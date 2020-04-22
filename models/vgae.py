import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv


class VGAE(nn.Module):
    def __init__(self, data, nhid1, nhid2, dropout):
        super(VGAE, self).__init__()
        nfeat = data.num_features
        self.gc1 = GCNConv(nfeat, nhid1)
        self.gc_mu = GCNConv(nhid1, nhid2)
        self.gc_logvar = GCNConv(nhid1, nhid2)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def encoder(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        mu, logvar = self.gc_mu(x, adj), self.gc_logvar(x, adj)
        return mu, logvar

    def reparametarize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decoder(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparametarize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar


def create_vgae_model(data, nhid1=32, nhid2=16, dropout=0.):
    model = VGAE(data, nhid1, nhid2, dropout)
    return model
