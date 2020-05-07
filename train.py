import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from numpy import mean, std
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, model, data, lr, epochs, verbose=True):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.data = data
        self.epochs = epochs
        self.verbose = verbose

        self.model.to(device).reset_parameters()
        self.data.to(device)

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

        with tqdm(range(1, self.epochs + 1)) as pbar:
            for _ in pbar:
                self.train()
                evals = self.evaluate()
                pbar.set_postfix(evals)

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
        error = model.recon_loss(data, output)
        return {'loss': float(loss), 'error': error}


class LinkPredTrainer(EmbeddingTrainer):
    def __init__(self, model, data, lr, epochs):
        super(LinkPredTrainer, self).__init__(model, data, lr, epochs)

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = model.loss_function(data, output)
        val_roc, val_ap = linkpred_score(output['mu'], data.val_edges, data.neg_val_edges)
        test_roc, test_ap = linkpred_score(output['mu'], data.test_edges, data.neg_test_edges)

        return {'loss': float(loss), 'val_roc': val_roc, 'val_ac': val_ap, 'test_roc': test_roc, 'test_ap': test_ap}


def linkpred_score(z, pos_edges, neg_edges):
    pos_score = torch.sigmoid(torch.sum(z[pos_edges[0]] * z[pos_edges[1]], dim=1))
    neg_score = torch.sigmoid(torch.sum(z[neg_edges[0]] * z[neg_edges[1]], dim=1))
    pred_score = torch.cat([pos_score, neg_score]).cpu().numpy()
    true_score = np.hstack([np.ones(pos_score.size(0)), np.zeros(neg_score.size(0))])
    roc_score = roc_auc_score(true_score, pred_score)
    ap_score = average_precision_score(true_score, pred_score)
    return roc_score, ap_score


class NodeClsTrainer(EmbeddingTrainer):
    def __init__(self):
        super(NodeClsTrainer, self).__init__()


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
