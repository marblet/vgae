import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from numpy import mean, std
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse = nn.MSELoss()


class Trainer(object):
    def __init__(self, model, data, lr, epochs):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.data = data
        self.epochs = epochs

        self.model.to(device).reset_parameters()
        self.data.to(device)

        self.mse = nn.MSELoss()

    def train(self):
        model, optimizer, data = self.model, self.optimizer, self.data
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function(data, output)
        loss.backward()
        optimizer.step()
        return loss

    def evaluate(self):
        raise NotImplementedError()

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(1, self.epochs + 1):
            self.train()
            evals = self.evaluate()
            print(epoch, evals)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            output = self.model(self.data)
        return output


class EmbeddingTrainer(Trainer):
    def __init__(self, model, data, lr, epochs):
        super(EmbeddingTrainer, self).__init__(model, data, lr, epochs)

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = model.loss_function(data, output)
        if 'recon_edges' in output:
            e = output['recon_edges']
            error = self.mse(output['adj_recon'], data.adjmat[e])
        else:
            error = self.mse(output['adj_recon'], data.adjmat)
        return {'loss': loss, 'error': error}


class ClusteringTrainer(Trainer):
    def __init__(self, model, data, lr, epochs, pretrain_epochs=100):
        super(ClusteringTrainer, self).__init__(model, data, lr, epochs)
        self.pretrain_epochs = pretrain_epochs

    def pretrain(self):
        model, optimizer, data = self.model, self.optimizer, self.data
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function_pretrain(data, output)
        loss.backward()
        optimizer.step()
        return loss

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = model.loss_function(data, output)
        error = self.mse(output['adj_recon'], data.adjmat)
        pred = model.classify(data)
        nmi = NMI(data.labels.cpu(), pred, average_method='arithmetic')

        return loss, error, nmi

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in tqdm(range(1, self.pretrain_epochs + 1)):
            self.pretrain()

        self.model.initialize_gmm(self.data)

        for epoch in range(1, self.epochs + 1):
            self.train()
            evals = self.evaluate()
            print(epoch, evals)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            output = self.model(self.data)
        return output
