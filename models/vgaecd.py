import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.mixture import GaussianMixture
from . import Encoder, Decoder, reparameterize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGAECD(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=8):
        super(VGAECD, self).__init__()
        self.K = data.num_classes
        self.encoder = Encoder(data, nhid, latent_dim)
        self.decoder = Decoder()

        self.pi = nn.Parameter(torch.FloatTensor(self.K))
        self.mu = nn.Parameter(torch.FloatTensor(self.K, latent_dim))
        self.logvar = nn.Parameter(torch.FloatTensor(self.K, latent_dim))

        N = data.features.size(0)
        E = data.adjmat.sum()
        pos_weight = (N * N - E) / E
        self.weight_mat = torch.where(data.adjmat > 0, pos_weight, torch.tensor(1.))
        self.norm = (N * N) / ((N * N - E) * 2)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.pi.data = torch.zeros_like(self.pi)
        self.mu.data = torch.zeros_like(self.mu)
        self.logvar.data = torch.zeros_like(self.logvar)

    def initialize_gmm(self, data):
        with torch.no_grad():
            mu, logvar = self.encoder(data)
            z = reparameterize(mu, logvar, self.training)
        z = z.cpu().detach().numpy()
        gmm = GaussianMixture(n_components=self.K, covariance_type='diag')
        gmm.fit(z)
        self.pi.data = torch.FloatTensor(gmm.weights_).to(device)
        self.mu.data = torch.FloatTensor(gmm.means_).to(device)
        self.logvar.data = torch.log(torch.FloatTensor(gmm.covariances_)).to(device)

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return self.norm * F.binary_cross_entropy(adj_recon, data.adjmat, weight=self.weight_mat)

    def loss_function_pretrain(self, data, output):
        recon_loss = self.recon_loss(data, output)
        mu, logvar = output['mu'], output['logvar']
        kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        return recon_loss + kl

    def loss_function(self, data, output):
        mu, logvar = output['mu'], output['logvar']
        recon_loss = self.recon_loss(data, output)

        z = reparameterize(mu, logvar, self.training).unsqueeze(1)
        h = z - self.mu
        h = torch.exp(-0.5 * torch.sum((h * h) / torch.exp(self.logvar), dim=2))
        weights = torch.softmax(self.pi, dim=0)

        h = h / torch.exp(torch.sum(0.5 * self.logvar, dim=1))
        p_z_given_c = h / (2 * math.pi)
        p_z_c = p_z_given_c * weights + 1e-9
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

        h = logvar.exp().pow(2).unsqueeze(1) + (mu.unsqueeze(1) - self.mu).pow(2)
        h = torch.sum(self.logvar + h / torch.exp(self.logvar), dim=2)
        com_loss = 0.5 * torch.sum(gamma * h) \
            - torch.sum(gamma * torch.log(weights + 1e-9)) \
            + torch.sum(gamma * torch.log(gamma + 1e-9)) \
            - 0.5 * torch.sum(1 + logvar)
        com_loss = com_loss / (data.num_nodes * data.num_nodes)
        return recon_loss + com_loss

    def classify(self, data, n_samples=8):
        with torch.no_grad():
            mu, logvar = self.encoder(data)
            weights = torch.softmax(self.pi, dim=0)
            z = torch.stack(
                [reparameterize(mu, logvar, self.training) for _ in range(n_samples)], dim=1)
            z = z.unsqueeze(2)
            h = z - self.mu
            h = torch.exp(-0.5 * torch.sum(h * h / torch.exp(self.logvar), dim=3))

            h = h / torch.exp(torch.sum(0.5 * self.logvar, dim=1))
            p_z_given_c = h / (2 * math.pi)
            p_z_c = p_z_given_c * weights
            y = p_z_c / torch.sum(p_z_c, dim=2, keepdim=True)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)
        return pred

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = reparameterize(mu, logvar, self.training)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'z': z, 'mu': mu, 'logvar': logvar}
