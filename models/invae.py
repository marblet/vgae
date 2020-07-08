import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Encoder, reparameterize


class INVAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.5):
        super(INVAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim)
        self.decoder = OmitDecoder(data, latent_dim, nhid, dropout)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def recon_loss(self, data, output):
        recon_adj = output['recon_adj']
        adj_loss = data.norm * F.binary_cross_entropy_with_logits(recon_adj, data.adjmat, pos_weight=data.pos_weight)
        recon_feat = output['recon_feat']
        feat_loss = F.mse_loss(recon_feat, data.features)
        return adj_loss + feat_loss

    def loss_function(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        return recon_loss + kl

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = reparameterize(mu, logvar, self.training)
        recon_adj, recon_feat = self.decoder(z)
        return {'recon_adj': recon_adj, 'recon_feat': recon_feat, 'z': z, 'mu': mu, 'logvar': logvar}


class OmitDecoder(nn.Module):
    def __init__(self, data, latent_dim, nhid, dropout):
        super(OmitDecoder, self).__init__()
        self.feat_fc1 = nn.Linear(latent_dim, nhid)
        self.feat_fc2 = nn.Linear(nhid, data.num_features)
        self.adj_fc1 = nn.Linear(latent_dim, nhid)
        self.adj_fc2 = nn.Linear(nhid, data.num_nodes)
        self.dropout = dropout

    def reset_parameters(self):
        self.feat_fc1.reset_parameters()
        self.feat_fc2.reset_parameters()

    def forward(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        a = F.relu(self.adj_fc1(z))
        a = F.dropout(a, p=self.dropout, training=self.training)
        recon_adj = self.adj_fc2(a)

        x = F.relu(self.feat_fc1(z))
        x = F.dropout(x, p=self.dropout, training=self.training)
        recon_feat = self.feat_fc2(x)
        return recon_adj, recon_feat
