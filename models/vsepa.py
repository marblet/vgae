import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv, Decoder, reparameterize, MLP, VMLP, ConcatDecoder
from utils import get_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VSEPA(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VSEPA, self).__init__()
        self.gcenc = VEncoder(data, nhid, latent_dim, dropout)
        self.mlpenc = VMLP(data.num_features, nhid, latent_dim, dropout)
        self.mlpdec = MLP(latent_dim, nhid, data.num_features, dropout, act=torch.sigmoid)
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
        feat_recon = self.mlpdec(zx)
        return {'adj_recon': adj_recon, 'feat_recon': feat_recon, 'pred': pred, 'z': z, 'mu_a': mu_a, 'logvar_a': logvar_a, 'mu_x': mu_x, 'logvar_x': logvar_x}


class VSEPAGRA(VSEPA):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VSEPAGRA, self).__init__(data, nhid, latent_dim, dropout)
        alpha = 0.95
        A = data.adjmat
        D = get_degree(data.edge_list)
        Dinv = 1 / D.float()
        self.gra = alpha * torch.matmul(torch.inverse(torch.eye(data.num_nodes) - alpha * torch.matmul(A, torch.diag(Dinv))), A)
        norm = self.gra.sum()
        self.gra = self.gra / norm * (data.num_nodes ** 2)
        self.gra = self.gra.to(device)

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        adj_recon_loss = F.mse_loss(adj_recon, self.gra)
        feat_recon = output['feat_recon']
        feat_recon_loss = F.mse_loss(feat_recon, data.features)
        return adj_recon_loss + feat_recon_loss


class VSEPATFN(VSEPA):
    def __init__(self, data, nhid=32, latent_dim=8, dropout=0.):
        super(VSEPATFN, self).__init__(data, nhid, latent_dim, dropout)
        self.mlpdec = MLP(latent_dim, nhid, data.num_features, dropout, act=torch.sigmoid)
        self.decoder = Decoder()

    def forward(self, data):
        mu_a, logvar_a = self.gcenc(data)
        mu_x, logvar_x = self.mlpenc(data.features)
        za = reparameterize(mu_a, logvar_a, self.training)
        zx = reparameterize(mu_x, logvar_x, self.training)
        za_ = torch.cat([za, torch.ones((data.num_nodes, 1), device=device)], dim=1)
        zx_ = torch.cat([zx, torch.ones((data.num_nodes, 1), device=device)], dim=1)
        z_tensor = torch.bmm(za_.unsqueeze(2), zx_.unsqueeze(1))
        z = torch.flatten(z_tensor, start_dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        feat_recon = self.mlpdec(zx)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'feat_recon': feat_recon, 'z': z,
                'mu_a': mu_a, 'logvar_a': logvar_a, 'mu_x': mu_x, 'logvar_x': logvar_x}


class VSEPACAT(VSEPA):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VSEPACAT, self).__init__(data, nhid, latent_dim, dropout)
        self.decoder = ConcatDecoder(data, latent_dim * 2, dropout)

    def recon_loss(self, data, output):
        adj_recon, recon_edges = output['adj_recon'], output['recon_edges']
        adj_recon_loss = F.binary_cross_entropy(adj_recon[:, 0], data.adjmat[recon_edges[0], recon_edges[1]])
        feat_recon = output['feat_recon']
        feat_recon_loss = F.mse_loss(feat_recon, data.features)
        return adj_recon_loss + feat_recon_loss

    def forward(self, data):
        mu_a, logvar_a = self.gcenc(data)
        mu_x, logvar_x = self.mlpenc(data.features)
        za = reparameterize(mu_a, logvar_a, self.training)
        zx = reparameterize(mu_x, logvar_x, self.training)
        z = torch.cat([za, zx], dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        pred = F.softmax(self.predlabel(z), dim=1)
        adj_recon, recon_edges = self.decoder(z, data)
        feat_recon = self.mlpdec(zx)
        return {'adj_recon': adj_recon, 'recon_edges': recon_edges, 'feat_recon': feat_recon, 'pred': pred,
                'z': z, 'mu_a': mu_a, 'logvar_a': logvar_a, 'mu_x': mu_x, 'logvar_x': logvar_x}


class VSEPACATTFN(VSEPACAT):
    def __init__(self, data, nhid=32, latent_dim=8, dropout=0.):
        super(VSEPACATTFN, self).__init__(data, nhid, latent_dim, dropout)
        self.decoder = ConcatDecoder(data, (latent_dim + 1) ** 2, dropout)

    def forward(self, data):
        mu_a, logvar_a = self.gcenc(data)
        mu_x, logvar_x = self.mlpenc(data.features)
        za = reparameterize(mu_a, logvar_a, self.training)
        zx = reparameterize(mu_x, logvar_x, self.training)
        z = torch.cat([za, zx], dim=1)
        za_ = torch.cat([za, torch.ones((data.num_nodes, 1), device=device)], dim=1)
        zx_ = torch.cat([zx, torch.ones((data.num_nodes, 1), device=device)], dim=1)
        z_tensor = torch.bmm(za_.unsqueeze(2), zx_.unsqueeze(1))
        z_flatten = torch.flatten(z_tensor, start_dim=1)
        z_flatten = F.dropout(z_flatten, p=self.dropout, training=self.training)
        feat_recon = self.mlpdec(zx)
        adj_recon, recon_edges = self.decoder(z_flatten, data)
        return {'adj_recon': adj_recon, 'recon_edges': recon_edges, 'feat_recon': feat_recon, 'z': z,
                'mu_a': mu_a, 'logvar_a': logvar_a, 'mu_x': mu_x, 'logvar_x': logvar_x}


class VEncoder(nn.Module):
    def __init__(self, data, nhid, latent_dim, dropout, bias=True):
        super(VEncoder, self).__init__()
        self.gc1 = GCNConv(data.num_features, nhid, bias)
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
