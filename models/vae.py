import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MLP, reparameterize


class VAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VAE, self).__init__()
        self.enc = VMLP(data.num_features, nhid, latent_dim, dropout)
        self.dec = MLP(latent_dim, nhid, data.num_features, dropout)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def recon_loss(self, data, output):
        feat_recon = output['feat_recon']
        return F.mse_loss(feat_recon, data.features)

    def loss_function(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return recon_loss + kl

    def forward(self, data):
        x = data.features
        mu, logvar = self.enc(x)
        z = reparameterize(mu, logvar, self.training)
        feat_recon = self.dec(z)
        return {'mu': mu, 'logvar': logvar, 'z': z, 'feat_recon': feat_recon}


class VMLP(nn.Module):
    def __init__(self, input_dim, nhid, latent_dim, dropout):
        super(VMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, nhid)
        self.fc_mu = nn.Linear(nhid, latent_dim)
        self.fc_logvar = nn.Linear(nhid, latent_dim)
        self.dropout = dropout

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar
