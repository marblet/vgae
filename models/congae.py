import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Encoder, reparameterize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CONVAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(CONVAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim)
        self.decoder = ConcatDecoder(data, latent_dim)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon, recon_edges = output['adj_recon'], output['recon_edges']
        return F.binary_cross_entropy(adj_recon[:, 0], data.adjmat[recon_edges[0], recon_edges[1]])

    def loss_function(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        return recon_loss + kl

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = reparameterize(mu, logvar, self.training)
        adj_recon, recon_edges = self.decoder(z, data)
        return {'adj_recon': adj_recon, 'recon_edges': recon_edges, 'z': z, 'mu': mu, 'logvar': logvar}


class ConcatDecoder(nn.Module):
    def __init__(self, data, latent_dim, dropout=0.):
        super(ConcatDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1)
        self.E = data.edge_list.size(1)
        self.negative_edges = torch.stack(torch.where(data.adjmat == 0))
        self.dropout = dropout

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def sample_negative_edges(self):
        idx = torch.randint(0, self.negative_edges.size(1), (self.E,))
        return self.negative_edges[:, idx]

    def forward(self, z, data):
        neg_edges = self.sample_negative_edges()
        recon_edges = torch.cat([data.edge_list, neg_edges], dim=1)
        source, target = recon_edges
        concat_z = torch.cat([z[source], z[target]], dim=1)
        concat_z = F.dropout(concat_z, p=self.dropout, training=self.training)
        z = self.fc1(concat_z)
        z = F.dropout(F.relu(z), p=self.dropout, training=self.training)
        adj_recon = self.fc2(z)
        adj_recon = torch.sigmoid(adj_recon)
        return adj_recon, recon_edges
