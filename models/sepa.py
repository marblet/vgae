import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv, Decoder, MLP, ConcatDecoder


class SEPA(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(SEPA, self).__init__()
        self.gcenc = GCEncoder(data, nhid, latent_dim, dropout, act=F.relu)
        self.mlpenc = MLP(data.num_features, nhid, latent_dim, dropout, act=F.relu)
        self.mlpdec = MLP(latent_dim, nhid, data.num_features, dropout)
        self.predlabel = nn.Linear(latent_dim * 2, data.num_classes)
        self.decoder = Decoder()
        self.dropout = dropout

    def reset_parameters(self):
        self.gcenc.reset_parameters()
        self.mlpenc.reset_parameters()
        self.mlpdec.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        adj_recon_loss = data.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=data.weight_mat)
        feat_recon = output['feat_recon']
        feat_recon_loss = F.mse_loss(feat_recon, data.features)
        return adj_recon_loss + feat_recon_loss

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        za = self.gcenc(data)
        zx = self.mlpenc(data.features)
        z = torch.cat([za, zx], dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        pred = F.softmax(self.predlabel(z), dim=1)
        adj_recon = self.decoder(z)
        feat_recon = torch.sigmoid(self.mlpdec(zx))
        return {'adj_recon': adj_recon, 'feat_recon': feat_recon, 'pred': pred, 'z': z}


class SEPACAT(SEPA):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(SEPACAT, self).__init__(data, nhid, latent_dim, dropout)
        self.decoder = ConcatDecoder(data, latent_dim * 2, dropout)

    def recon_loss(self, data, output):
        adj_recon, recon_edges = output['adj_recon'], output['recon_edges']
        adj_recon_loss = F.binary_cross_entropy(adj_recon[:, 0], data.adjmat[recon_edges[0], recon_edges[1]])
        feat_recon = output['feat_recon']
        feat_recon_loss = F.mse_loss(feat_recon, data.features)
        return adj_recon_loss + feat_recon_loss

    def forward(self, data):
        za = self.gcenc(data)
        zx = self.mlpenc(data.features)
        z = torch.cat([za, zx], dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        pred = F.softmax(self.predlabel(z), dim=1)
        adj_recon, recon_edges = self.decoder(z, data)
        feat_recon = torch.sigmoid(self.mlpdec(zx))
        return {'adj_recon': adj_recon, 'recon_edges': recon_edges, 'feat_recon': feat_recon, 'pred': pred, 'z': z}


class GCEncoder(nn.Module):
    def __init__(self, data, nhid, latent_dim, dropout, act=lambda x: x):
        super(GCEncoder, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid)
        self.gc2 = GCNConv(nhid, latent_dim)
        self.dropout = dropout
        self.act = act

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return self.act(x)
