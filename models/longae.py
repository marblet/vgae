import torch
import torch.nn as nn
import torch.nn.functional as F


class LoNGAE(nn.Module):
    def __init__(self, data, nhid=256, latent_dim=128):
        super(LoNGAE, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(data.num_nodes + data.num_features, nhid))
        self.W2 = nn.Parameter(torch.FloatTensor(nhid, latent_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(nhid))
        self.b2 = nn.Parameter(torch.FloatTensor(latent_dim))
        self.b3 = nn.Parameter(torch.FloatTensor(nhid))
        self.b4 = nn.Parameter(torch.FloatTensor(data.num_nodes + data.num_features))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1, gain=1.414)
        nn.init.xavier_uniform_(self.W2, gain=1.414)
        self.b1.data.fill_(0)
        self.b2.data.fill_(0)
        self.b3.data.fill_(0)
        self.b4.data.fill_(0)

    def recon_loss(self, data, output):
        true_x = torch.cat([data.adjmat, data.features], dim=1)
        return F.binary_cross_entropy(output['recon_x'], true_x)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x = torch.cat([data.adjmat, data.features], dim=1)
        x = F.relu(torch.matmul(x, self.W1) + self.b1)
        z = F.relu(torch.matmul(x, self.W2) + self.b2)
        x = F.relu(torch.matmul(z, self.W2.t()) + self.b3)
        x = torch.matmul(x, self.W1.t()) + self.b4
        return {'z': z, 'recon_x': torch.sigmoid(x)}
