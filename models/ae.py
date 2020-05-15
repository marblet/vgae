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


class AEShare(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(AEShare, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(data.num_features, nhid))
        self.W2 = nn.Parameter(torch.FloatTensor(nhid, latent_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(nhid))
        self.b2 = nn.Parameter(torch.FloatTensor(latent_dim))
        self.b3 = nn.Parameter(torch.FloatTensor(nhid))
        self.b4 = nn.Parameter(torch.FloatTensor(data.num_features))
        self.dropout = dropout

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1, gain=1.414)
        nn.init.xavier_uniform_(self.W2, gain=1.414)
        self.b1.data.fill_(0)
        self.b2.data.fill_(0)
        self.b3.data.fill_(0)
        self.b4.data.fill_(0)

    def recon_loss(self, data, output):
        return F.mse_loss(output['feat_recon'], data.features)

    def loss_function(self, data, output):
        return self.recon_loss(data, output)

    def forward(self, data):
        x = data.features
        x = F.relu(torch.matmul(x, self.W1) + self.b1)
        z = F.relu(torch.matmul(x, self.W2) + self.b2)
        x = F.relu(torch.matmul(z, self.W2.t()) + self.b3)
        feat_recon = torch.matmul(x, self.W1.t()) + self.b4
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
