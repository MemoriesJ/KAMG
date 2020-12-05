import torch
import torch.nn as nn
import torch.nn.init as init
import torch.sparse


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, act=torch.relu_, featureless=False, dropout=0.0):

        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.featureless = featureless
        self.act = act
        self.reset_parameter()

    def reset_parameter(self):
        init.xavier_uniform_(self.weight)

    def forward(self, inp, adj):

        if self.dropout is not None:
            inp = self.dropout(inp)

        if not self.featureless:
            inp = torch.matmul(inp, self.weight)
        else:
            inp = self.weight

        inp = torch.matmul(adj, inp)

        if self.bias is not None:
            inp += self.bias

        if self.act is not None:
            inp = self.act(inp)

        return inp


class MultiGraphConvolution(nn.Module):

    def __init__(self, n_adj, in_features, out_features, bias=False, act=torch.relu, featureless=False, dropout=0.0):
        """
        used for vectorized computing for multi-adj GCN
        """
        assert n_adj >= 2

        super(MultiGraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.empty(n_adj, in_features, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.act = act
        self.featureless = featureless
        self.reset_parameter()

    def reset_parameter(self):
        init.xavier_uniform_(self.weight)

    def forward(self, inp, adj):

        """inp.ndim in [2, 3]
        adj.ndim == 3
        adj.shape[0] == self.n_adj
        """

        if self.dropout is not None:
            inp = self.dropout(inp)

        # adj = torch.unsqueeze(adj, dim=0) if inp.ndim == adj.ndim else adj
        # inp = torch.unsqueeze(inp, dim=inp.ndim - 2)

        if not self.featureless:
            inp = torch.matmul(inp, self.weight)
        else:
            inp = self.weight

        inp = torch.matmul(adj, inp)

        if self.bias is not None:
            inp += self.bias

        if self.act is not None:
            inp = self.act(inp)

        return inp
