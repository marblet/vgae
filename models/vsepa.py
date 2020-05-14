import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv, Decoder, reparameterize, MLP


class VSEPA(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VSEPA, self).__init__()
        self.gcenc = VEncoder(data.num_features, nhid, latent_dim, dropout)
        self.mlpenc = VMLP(data.num_features, nhid, latent_dim, dropout)
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
        recon_loss = self.recon_loss(data, output)
        mu_a, logvar_a = output['mu_a'], output['logvar_a']
        mu_x, logvar_x = output['mu_x'], output['logvar_x']
        kl_a = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar_a - mu_a.pow(2) - logvar_a.exp().pow(2), 1))
        kl_x = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar_x - mu_x.pow(2) - logvar_x.exp().pow(2), 1))
        return recon_loss + kl_a + kl_x

    def forward(self, data):
        mu_a, logvar_a = self.gcenc(data)
        mu_x, logvar_x = self.mlpenc(data.features)
        za = reparameterize(mu_a, logvar_a, self.training)
        zx = reparameterize(mu_x, logvar_x, self.training)
        z = torch.cat([za, zx], dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        pred = F.softmax(self.predlabel(z), dim=1)
        adj_recon = self.decoder(z)
        feat_recon = torch.sigmoid(self.mlpdec(zx))
        return {'adj_recon': adj_recon, 'feat_recon': feat_recon, 'pred': pred, 'z': z, 'mu_a': mu_a, 'logvar_a': logvar_a, 'mu_x': mu_x, 'logvar_x': logvar_x}


class VEncoder(nn.Module):
    def __init__(self, nfeat, nhid, latent_dim, dropout, bias=True):
        super(VEncoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid, bias)
        self.gc_mu = GCNConv(nhid, latent_dim, bias)
        self.gc_logvar = GCNConv(nhid, latent_dim, bias)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu, logvar = self.gc_mu(x, adj), self.gc_logvar(x, adj)
        return mu, logvar


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
