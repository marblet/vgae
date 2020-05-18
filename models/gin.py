import torch
import torch.nn as nn


class GINConv(nn.Module):
    def __init__(self, data, in_features, out_features, eps=True):
        super(GINConv, self).__init__()
        indices = data.adjmat.nonzero().t()
        values = torch.ones(indices.size(1))
        self.ginadj = torch.sparse.FloatTensor(indices, values)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if eps:
            self.eps = nn.Parameter(torch.FloatTensor(1))
        else:
            self.register_parameter('eps', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.eps is not None:
            self.eps.data.fill_(0)

    def forward(self, x, adj):
        wx = torch.matmul(x, self.weight)
        x = torch.spmm(self.ginadj, wx)
        if self.eps is not None:
            x += self.eps * wx
        return x
