import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv, Decoder


class GAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(GAE, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid)
        self.gc2 = GCNConv(nhid, latent_dim)
        self.decoder = Decoder()

        N = data.features.size(0)
        E = data.edge_list.size(1)
        pos_weight = torch.tensor((N * N) / E - 1)
        self.weight_mat = torch.where(data.adjmat > 0, pos_weight, torch.tensor(1.))
        self.norm = (N * N) / ((N * N - E) * 2)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return self.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=self.weight_mat)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x, adj = data.features, data.adj
        x = self.gc1(x, adj)
        z = self.gc2(x, adj)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'z': z}
