import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_degree
from . import GCNConv


def laplacian_sharpening(data):
    deg = get_degree(data.edge_list) + 1
    source, target = data.edge_list
    weight = - torch.ones(data.edge_list.size(1))
    weight += 3 * (source == target)
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    v = deg_inv_sqrt[source] * weight * deg_inv_sqrt[target]
    lap_sharp = torch.sparse.FloatTensor(data.edge_list, v)
    return lap_sharp


class GALA(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(GALA, self).__init__()
        self.enc1 = GCNConv(data.num_features, nhid)
        self.enc2 = GCNConv(nhid, latent_dim)
        self.dec1 = GCNConv(latent_dim, nhid)
        self.dec2 = GCNConv(nhid, data.num_features)
        self.lap_sharp = laplacian_sharpening(data)

    def reset_parameters(self):
        self.enc1.reset_parameters()
        self.enc2.reset_parameters()
        self.dec1.reset_parameters()
        self.dec2.reset_parameters()

    def recon_loss(self, data, output):
        x_recon = output['x_recon']
        return F.mse_loss(x_recon, data.features)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.relu(self.enc1(x, adj))
        z = F.relu(self.enc2(x, adj))
        x = F.relu(self.dec1(z, self.lap_sharp))
        x_recon = F.relu(self.dec2(x, self.lap_sharp))
        return {'x_recon': x_recon, 'z': z}
