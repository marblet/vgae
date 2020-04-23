import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from numpy import mean, std
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse = nn.MSELoss()


def train(model, optimizer, data, norm, pos_weight, pretrain=False):
    model.train()
    optimizer.zero_grad()
    z, mu, logvar = model(data)
    loss = model.loss_function(data, z, mu, logvar, norm, pos_weight, pretrain)
    loss.backward()
    optimizer.step()
    return loss


def evaluate(model, data, norm, pos_weight):
    model.eval()

    with torch.no_grad():
        z, mu, logvar = model(data)

    loss = model.loss_function(data, z, mu, logvar, norm, pos_weight)
    error = mse(z, data.adjmat)
    pred = model.classify(data)
    nmi = NMI(data.labels.cpu(), pred, average_method='arithmetic')

    return loss, error, nmi


def run(data, model, lr, epochs=200, pretrain=100, niter=1, verbose=False):
    # for GPU
    data.to(device)

    N = data.num_nodes
    E = data.edge_list.size(1)
    pw = torch.tensor((N * N) / E - 1)
    ones = torch.ones_like(data.adjmat)
    pos_weight = torch.where(data.adjmat > 0, pw * ones, ones)
    norm = (N * N) / ((N * N - E) * 2)

    for _ in tqdm(range(niter)):
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("Pretrain")
        for epoch in range(1, pretrain + 1):
            l = train(model, optimizer, data, norm, pos_weight, True)
            print(l)

        print("Initialize GMM parameters")
        model.initialize_gmm(data)

        print("Train model")
        for epoch in range(1, epochs + 1):
            train(model, optimizer, data, norm, pos_weight)
            evals = evaluate(model, data, norm, pos_weight)
            print(epoch, evals)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        z, mu, logvar = model(data)

    return z
