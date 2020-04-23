import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from numpy import mean, std
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse = nn.MSELoss()


def train(model, optimizer, data, pretrain=False):
    model.train()
    optimizer.zero_grad()
    z, mu, logvar = model(data)
    loss = model.loss_function(data, z, mu, logvar, pretrain)
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        z, mu, logvar = model(data)

    loss = model.loss_function(data, z, mu, logvar)
    error = mse(z, data.adjmat)
    pred = model.classify(data)
    nmi = NMI(data.labels, pred, average_method='arithmetic')

    return loss, error, nmi


def run(data, model, lr, weight_decay, epochs=200, pretrain=100, niter=1, verbose=False):
    # for GPU
    data.to(device)

    for _ in tqdm(range(niter)):
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(1, pretrain + 1):
            train(model, optimizer, data, True)

        model.initialize_gmm(data)

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            evals = evaluate(model, data)
            print(epoch, evals)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        z, mu, logvar = model(data)

    return mu
