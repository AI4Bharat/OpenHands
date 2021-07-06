import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
from .graph_utils import SpatialGraph


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_residual=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.einsum("bij,jk -> bik", x, self.weight)
        adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        output = torch.einsum("bij, bjk -> bik", adj, support)

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)

        if self.use_residual:
            output = output + x

        return output


class GCNModel(nn.Module):
    def __init__(self, num_points, inward_edges, in_channels, num_layers=2):
        super().__init__()

        self.num_points = num_points
        self.num_layers = num_layers

        modules = []
        for i in range(num_layers):
            modules.append(GCNLayer(in_channels, in_channels))

        self.gcn = nn.ModuleList(modules)
        self.register_buffer(
            "adj",
            torch.FloatTensor(
                SpatialGraph(num_nodes=num_points, inward_edges=inward_edges).A.sum(
                    axis=0
                )
            ),
        )

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(-1)
        x = x.permute(0, 2, 3, 1)  # B, C, T, V ->B, T, V, C

        B, T, V, C = x.shape
        x = x.reshape(B * T, V, C)
        for i, gcn_layer in enumerate(self.gcn):
            x = gcn_layer(x, self.adj)
        x = x.reshape(B, T, V, C)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # B, T, V, C -> B, C, T, V, M
        return x