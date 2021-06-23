import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from omegaconf import OmegaConf

#https://github.com/jackyjsy/CVPR21Chal-SLR

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, num_nodes, inward_edges, strategy="spatial"):

        self.num_nodes = num_nodes
        self.strategy = strategy
        self.self_edges = [(i, i) for i in range(num_nodes)]
        self.inward_edges = inward_edges
        self.outward_edges = [(j, i) for (i, j) in self.inward_edges]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        if self.strategy == "spatial":
            return get_spatial_graph(
                self.num_nodes, self.self_edges, self.inward_edges, self.outward_edges
            )
        else:
            raise ValueError()


class DropGraphTemporal(nn.Module):
    def __init__(self, block_size=7):
        super(DropGraphTemporal, self).__init__()
        self.block_size = block_size

    def forward(self, x, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return x

        n, c, t, v = x.size()

        input_abs = torch.mean(torch.mean(torch.abs(x), dim=3), dim=1).detach()
        input_abs = (input_abs / torch.sum(input_abs) * input_abs.numel()).view(n, 1, t)
        gamma = (1.0 - self.keep_prob) / self.block_size
        input1 = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1, c * v, 1)
        m_sum = F.max_pool1d(
            M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2
        )
        mask = (1 - m_sum).to(device=m_sum.device, dtype=m_sum.dtype)
        return (
            (input1 * mask * mask.numel() / mask.sum())
            .view(n, c, v, t)
            .permute(0, 1, 3, 2)
        )


class DropGraphSpatial(nn.Module):
    def __init__(self, num_points, drop_size):
        super(DropGraphSpatial, self).__init__()
        self.drop_size = drop_size
        self.num_points = num_points

    def forward(self, x, keep_prob, A):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return x

        n, c, t, v = x.size()
        input_abs = torch.mean(torch.mean(torch.abs(x), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()

        gamma = (1.0 - self.keep_prob) / (1 + self.drop_size)
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(
            device=x.device, dtype=x.dtype
        )
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0

        mask = (1 - M).view(n, 1, 1, self.num_points)
        return x * mask * mask.numel() / mask.sum()


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def find_drop_size(num_nodes, num_edges, K=1):
    B_sum = 0
    for i in range(1, K + 1):
        B_sum += (2 * num_edges / num_nodes) * math.pow(
            (2 * num_edges / num_nodes) - 1, i - 1
        )
    return B_sum


class TCNUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        use_drop=True,
        drop_size=1.92,
        num_points=25,
        block_size=41,
    ):
        super(TCNUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.use_drop = use_drop
        if use_drop:
            self.dropS = DropGraphSpatial(num_points=num_points, drop_size=drop_size)
            self.dropT = DropGraphTemporal(block_size=block_size)

    def forward(self, x, keep_prob=None, A=None):
        x = self.bn(self.conv(x))
        if self.use_drop:
            x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class DecoupledGCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_points, num_subset=3):
        super(DecoupledGCNUnit, self).__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.num_subset = num_subset

        self.decoupled_A = nn.Parameter(
            torch.tensor(
                np.reshape(A, [3, 1, num_points, num_points]), dtype=torch.float32
            ).repeat(1, groups, 1, 1),
            requires_grad=True,
        )

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.linear_weight = nn.Parameter(
            torch.zeros(in_channels, out_channels * num_subset), requires_grad=True
        )
        self.linear_bias = nn.Parameter(
            torch.zeros(1, out_channels * num_subset, 1, 1), requires_grad=True
        )

        self.eye_list = nn.Parameter(
            torch.stack([torch.eye(num_points) for _ in range(out_channels)]),
            requires_grad=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        bn_init(self.bn, 1e-6)
        nn.init.normal_(
            self.linear_weight, 0, math.sqrt(0.5 / (out_channels * num_subset))
        )
        nn.init.constant_(self.linear_bias, 1e-6)

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_points, self.num_points)
        D_list = torch.sum(A, 1).view(c, 1, self.num_points)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eye_list * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_adj = self.decoupled_A.repeat(1, self.out_channels // self.groups, 1, 1)
        normed_adj = torch.cat(
            [
                self.norm(learn_adj[0:1, ...]),
                self.norm(learn_adj[1:2, ...]),
                self.norm(learn_adj[2:3, ...]),
            ],
            0,
        )

        x = torch.einsum("nctw,cd->ndtw", (x0, self.linear_weight)).contiguous()
        x = x + self.linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum("nkctv,kcvw->nctw", (x, normed_adj))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class DecoupledGCN_TCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        groups,
        num_points,
        block_size,
        drop_size,
        stride=1,
        residual=True,
        use_attention=True,
    ):
        super(DecoupledGCN_TCN_unit, self).__init__()

        num_joints = A.shape[-1]
        self.gcn1 = DecoupledGCNUnit(in_channels, out_channels, A, groups, num_points)
        self.tcn1 = TCNUnit(
            out_channels,
            out_channels,
            stride=stride,
            num_points=num_points,
            drop_size=drop_size,
        )
        self.relu = nn.ReLU()
        self.A = nn.Parameter(
            torch.tensor(
                np.sum(
                    np.reshape(A.astype(np.float32), [3, num_points, num_points]), axis=0
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TCNUnit(
                in_channels, out_channels, kernel_size=1, stride=stride, use_drop=False
            )

        self.drop_spatial = DropGraphSpatial(num_points=num_points, drop_size=drop_size)
        self.drop_temporal = DropGraphTemporal(block_size=block_size)

        self.use_attention = use_attention
        if self.use_attention:
            self.sigmoid = nn.Sigmoid()

            # Temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            # Spatial Attention
            ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # Channel Attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x, keep_prob):
        y = self.gcn1(x)
        if self.use_attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.residual(x)
        x_skip = self.drop_spatial(x_skip, keep_prob, self.A)
        x_skip = self.drop_temporal(x_skip, keep_prob)
        return self.relu(y + x_skip)


class DecoupledGCN(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_points=27,
        inward_edges=[],
        groups=8,
        block_size=41,
    ):
        super(DecoupledGCN, self).__init__()
       
        self.graph = Graph(num_points, inward_edges)
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(in_channels * num_points)

        drop_size = find_drop_size(self.graph.num_nodes, len(self.graph.inward_edges))
        self.l1 = DecoupledGCN_TCN_unit(
            in_channels,
            64,
            A,
            groups,
            num_points,
            block_size,
            drop_size=drop_size,
            residual=False,
        )
        self.l2 = DecoupledGCN_TCN_unit(
            64, 64, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.l3 = DecoupledGCN_TCN_unit(
            64, 64, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.l4 = DecoupledGCN_TCN_unit(
            64, 64, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.l5 = DecoupledGCN_TCN_unit(
            64, 128, A, groups, num_points, block_size, drop_size=drop_size, stride=2
        )
        self.l6 = DecoupledGCN_TCN_unit(
            128, 128, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.l7 = DecoupledGCN_TCN_unit(
            128, 128, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.l8 = DecoupledGCN_TCN_unit(
            128, 256, A, groups, num_points, block_size, drop_size=drop_size, stride=2
        )
        self.l9 = DecoupledGCN_TCN_unit(
            256, 256, A, groups, num_points, block_size, drop_size=drop_size
        )
        self.n_out_features = 256
        self.l10 = DecoupledGCN_TCN_unit(
            256, self.n_out_features, A, groups, num_points, block_size, drop_size=drop_size
        )

        bn_init(self.data_bn, 1)
        # self.fc = nn.Linear(256, num_class)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x, keep_prob=0.9):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )
        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)
        # N*M,C,T,V
        c_new = x.size(1)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        return x.mean(3).mean(1)
        # return self.fc(x)
