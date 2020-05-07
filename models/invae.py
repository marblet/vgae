import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import GCNConv, Encoder, reparameterize


class INVAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(INVAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim)
        self.decoder = InverseDecoder(data, latent_dim, nhid)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return data.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=data.weight_mat)

    def loss_function(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        return recon_loss + kl

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = reparameterize(mu, logvar, self.training)
        recon_feat = self.decoder(z)
        return {'recon_feat': recon_feat, 'z': z, 'mu': mu, 'logvar': logvar}


def inverse(adj):
    adj = adj.to_dense().numpy()
    return torch.tensor(np.linalg.pinv(adj))


class DenseGCNConv(GCNConv):
    def __init__(self, in_features, out_features):
        super(DenseGCNConv, self).__init__(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        if self.bias is not None:
            x += self.bias
        return x


class InverseDecoder(nn.Module):
    def __init__(self, data, latent_dim, nhid):
        super(InverseDecoder, self).__init__()
        self.gc1 = DenseGCNConv(latent_dim, nhid)
        self.gc2 = DenseGCNConv(nhid, data.num_features)
        self.inv_adj = inverse(data.adj)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, z):
        z = F.relu(self.gc1(z, self.inv_adj))
        recon_x = self.gc2(z, self.inv_adj)
        return torch.sigmoid(recon_x)
