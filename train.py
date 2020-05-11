import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from numpy import mean, std
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, model, data, lr, weight_decay, epochs, verbose=True):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    def __init__(self, model, data, lr, weigh_decay, epochs):
        super(EmbeddingTrainer, self).__init__(model, data, lr, weigh_decay, epochs)

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = model.loss_function(data, output)
        error = model.recon_loss(data, output)
        return {'loss': float(loss), 'error': float(error)}


class LinkPredTrainer(EmbeddingTrainer):
    def __init__(self, model, data, lr, weight_decay, epochs):
        super(LinkPredTrainer, self).__init__(model, data, lr, weight_decay, epochs)

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = model.loss_function(data, output)
        val_roc, val_ap = linkpred_score(output['z'], data.val_edges, data.neg_val_edges)
        test_roc, test_ap = linkpred_score(output['z'], data.test_edges, data.neg_test_edges)

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
    def __init__(self, model, data, lr, weight_decay, epochs):
        super(NodeClsTrainer, self).__init__(model, data, lr, weight_decay, epochs)

    def test(self, embed):
        nmi_scores = []
        ami_scores = []
        ari_scores = []
        true_labels = self.data.labels.cpu()
        for _ in range(10):
            pred = KMeans(n_clusters=self.data.num_classes).fit_predict(embed)
            nmi = NMI(true_labels, pred, average_method='arithmetic')
            ami = AMI(true_labels, pred, average_method='arithmetic')
            ari = ARI(true_labels, pred)
            nmi_scores.append(nmi)
            ami_scores.append(ami)
            ari_scores.append(ari)
        print("NMI", mean(nmi_scores), std(nmi_scores))
        print("AMI", mean(ami_scores), std(ami_scores))
        print("ARI", mean(ari_scores), std(ari_scores))

    def run(self):
        output = super().run()
        embed = output['z'].cpu().detach().numpy()
        self.test(embed)
        return output


class SemiNodeClsTrainer(EmbeddingTrainer):
    def __init__(self, model, data, lr, weight_decay, epochs):
        super(SemiNodeClsTrainer, self).__init__(model, data, lr, weight_decay, epochs)

    def train(self):
        model, optimizer, data = self.model, self.optimizer, self.data
        model.train()
        optimizer.zero_grad()
        output = model(data)
        model_loss = model.loss_function(data, output)
        pred_loss = F.nll_loss(output['pred'][data.train_mask], data.labels[data.train_mask])
        loss = model_loss + pred_loss
        loss.backward()
        optimizer.step()

    def evaluate(self):
        model, data = self.model, self.data
        model.eval()

        with torch.no_grad():
            output = model(data)

        evals = {}
        for key in ['train', 'val', 'test']:
            if key == 'train':
                mask = data.train_mask
            elif key == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            pred = output['pred'][mask].max(dim=1)[1]
            acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

            evals['{}_acc'.format(key)] = acc

        return evals


class ClusteringTrainer(Trainer):
    def __init__(self, model, data, lr, weight_decay, epochs, pretrain_epochs=100):
        super(ClusteringTrainer, self).__init__(model, data, lr, weight_decay, epochs)
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
