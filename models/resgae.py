import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCNConv, Decoder


class RESGAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16):
        super(RESGAE, self).__init__()
        self.enc1 = GCNConv(data.num_features, nhid)
        self.enc2 = GCNConv(nhid, latent_dim)
        self.enc3 = GCNConv(latent_dim, data.num_classes)
        self.dec1 = GCNConv(data.num_classes, latent_dim)
        self.dec2 = GCNConv(latent_dim, nhid)
        self.dec3 = GCNConv(nhid, data.num_features)

    def reset_parameters(self):
        self.enc1.reset_parameters()
        self.enc2.reset_parameters()
        self.enc3.reset_parameters()
        self.dec1.reset_parameters()
        self.dec2.reset_parameters()
        self.dec3.reset_parameters()

    def recon_loss(self, data, output):
        feat_recon = output['feat_recon']
        return F.mse_loss(feat_recon, data.features)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x, adj = data.features, data.adj
        x1 = F.tanh(self.enc1(x, adj))
        x2 = F.tanh(self.enc2(x1, adj))
        z = F.tanh(self.enc3(x2, adj))
        z2 = F.tanh(self.dec1(z, adj)) + x2
        z1 = F.tanh(self.dec2(z2, adj)) + x1
        feat_recon = torch.sigmoid(self.dec3(z1, adj))
        return {'feat_recon': feat_recon, 'z': z, 'pred': F.softmax(z, dim=1)}
