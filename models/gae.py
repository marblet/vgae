import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv, Decoder


class GAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(GAE, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid, bias=False)
        self.gc2 = GCNConv(nhid, latent_dim, bias=False)
        self.decoder = Decoder()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return data.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=data.weight_mat)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x, adj = data.features, data.adj
        x = self.gc1(x, adj)
        z = self.gc2(x, adj)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'z': z}
