import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from numpy import mean, std
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse = nn.MSELoss()


def loss_function(data, z, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(z, data.adjmat)
    kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
    return recon_loss + kl


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z, mu, logvar = model(data)
    loss = loss_function(data, z, mu, logvar)
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        z, mu, logvar = model(data)

    loss = loss_function(data, z, mu, logvar)
    error = mse(z, data.adjmat)

    return loss, error


def run(data, model, lr, weight_decay, epochs=200, niter=1, verbose=False):
    # for GPU
    data.to(device)

    for _ in tqdm(range(niter)):
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            evals = evaluate(model, data)
            print(epoch, evals)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        z, mu, logvar = model(data)

    return mu
