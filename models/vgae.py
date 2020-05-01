import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv


class VGAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(VGAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim)
        self.decoder = Decoder()

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def loss_function(self, data, z, mu, logvar, norm, pos_weight, pretrain=False):
        recon_loss = norm * F.binary_cross_entropy(z, data.adjmat, weight=pos_weight)
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        return recon_loss + kl

    def classify(self, data):
        return [0] * data.num_nodes

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return {'z': z, 'mu': mu, 'logvar': logvar}


class Encoder(nn.Module):
    def __init__(self, data, nhid, latent_dim):
        super(Encoder, self).__init__()
        nfeat = data.num_features
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc_mu = GCNConv(nhid, latent_dim)
        self.gc_logvar = GCNConv(nhid, latent_dim)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.relu(self.gc1(x, adj))
        mu, logvar = self.gc_mu(x, adj), self.gc_logvar(x, adj)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z):
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj
