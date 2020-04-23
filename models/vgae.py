import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv


class VGAE(nn.Module):
    def __init__(self, data, nhid, latent_dim, dropout):
        super(VGAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim, dropout)
        self.decoder = Decoder(dropout)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def loss_function(self, data, z, mu, logvar):
        recon_loss = F.binary_cross_entropy_with_logits(z, data.adjmat)
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
        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(self, data, nhid, latent_dim, dropout):
        super(Encoder, self).__init__()
        nfeat = data.num_features
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc_mu = GCNConv(nhid, latent_dim)
        self.gc_logvar = GCNConv(nhid, latent_dim)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        mu, logvar = self.gc_mu(x, adj), self.gc_logvar(x, adj)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj


def create_vgae_model(data, nhid=32, latent_dim=16, dropout=0.):
    model = VGAE(data, nhid, latent_dim, dropout)
    return model
