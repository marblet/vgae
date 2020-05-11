import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv, Decoder


class SEPA(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(SEPA, self).__init__()
        self.gcenc = GCEncoder(data.num_features, nhid, latent_dim, dropout=0.)
        self.mlpenc = MLP(data.num_features, nhid, latent_dim)
        self.mlpdec = MLP(latent_dim, nhid, data.num_features)
        self.decoder = Decoder()

    def reset_parameters(self):
        self.gcenc.reset_parameters()
        self.mlpenc.reset_parameters()
        self.mlpdec.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return data.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=data.weight_mat)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        za = self.gcenc(data)
        zx = self.mlpenc(data.features)
        z = torch.cat([za, zx], dim=1)
        adj_recon = self.decoder(z)
        feat_recon = self.mlpdec(zx)
        return {'adj_recon': adj_recon, 'feat_recon': feat_recon, 'z': z}


class GCEncoder(nn.Module):
    def __init__(self, input_dim, nhid, latent_dim, dropout=0.):
        super(GCEncoder, self).__init__()
        self.gc1 = GCNConv(input_dim, nhid)
        self.gc2 = GCNConv(nhid, latent_dim)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, nhid, latent_dim, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, nhid)
        self.fc2 = nn.Linear(nhid, latent_dim)
        self.dropout = dropout

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return x
