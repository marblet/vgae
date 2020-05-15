import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(AE, self).__init__()
        self.enc = MLP(data.num_features, nhid, latent_dim, dropout)
        self.dec = MLP(latent_dim, nhid, data.num_features, dropout)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def recon_loss(self, data, output):
        return F.mse_loss(output['feat_recon'], data.features)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x = data.features
        z = self.enc(x)
        feat_recon = torch.sigmoid(self.dec(z))
        return {'z': z, 'feat_recon': feat_recon}


class MLP(nn.Module):
    def __init__(self, input_dim, nhid, latent_dim, dropout):
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
        x = self.fc2(x)
        return x
