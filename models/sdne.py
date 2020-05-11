import torch
import torch.nn as nn
import torch.nn.functional as F


class SDNE(nn.Module):
    def __init__(self, data, nhid=200, latent_dim=100):
        super(SDNE, self).__init__()
        self.enc1 = nn.Linear(data.num_nodes, nhid)
        self.enc2 = nn.Linear(nhid, latent_dim)
        self.dec1 = nn.Linear(latent_dim, nhid)
        self.dec2 = nn.Linear(nhid, data.num_nodes)

    def reset_parameters(self):
        self.enc1.reset_parameters()
        self.enc2.reset_parameters()
        self.dec1.reset_parameters()
        self.dec2.reset_parameters()

    def recon_loss(self, data, output):
        adj_recon = output['adj_recon']
        return torch.sum((adj_recon - data.adjmat) * data.weight_mat).pow(2)

    def loss_function(self, data, output):
        l_2nd = self.recon_loss(data, output)
        z = output['z']
        source, target = data.edge_list
        zi, zj = z[source], z[target]
        l_1st = torch.sum((zi - zj).pow(2))
        return l_2nd + 0.1 * l_1st

    def forward(self, data):
        adjmat = data.adjmat
        x = F.relu(self.enc1(adjmat))
        z = F.relu(self.enc2(x))
        x = F.relu(self.dec1(z))
        recon_adjmat = torch.sigmoid(self.dec2(x))
        return {'z': z, 'adj_recon': recon_adjmat}
