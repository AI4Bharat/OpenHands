import torch
from torch import nn

import math

#Adopted from: https://github.com/microsoft/SGN

class norm_data(nn.Module):
    def __init__(self, n_joints, dim=3):
        super(norm_data, self).__init__()
        self.bn = nn.BatchNorm1d(dim * n_joints)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class embed(nn.Module):
    def __init__(self, n_joints, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        self.cnn = nn.Sequential(
            norm_data(n_joints, dim) if norm else nn.Identity(),
            cnn1x1(dim, 64, bias=bias),
            nn.ReLU(),
            cnn1x1(64, dim1, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class SGN(nn.Module):
    """
    SGN model proposed in 
    `Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition
    <https://arxiv.org/pdf/1904.01189.pdf>`_

    Note:
        The model supports inputs only with fixed number of frames.

    Args:
        n_frames (int): Number of frames in the input sequence.
        num_points (int): Number of spatial points in a graph.
        in_channels (int): Number of channels in the input data. Default: 2.
        bias (bool): Whether to use bias or not. Default: ``True``.
    """

    def __init__(self, n_frames, num_points, in_channels=2, bias=True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.n_frames = n_frames
        self.n_joints = num_points

        self.register_buffer(
            "spa_oh", self.one_hot(1, self.n_joints, self.n_frames).permute(0, 3, 2, 1)
        )
        self.register_buffer(
            "tem_oh", self.one_hot(1, self.n_frames, self.n_joints).permute(0, 3, 1, 2)
        )

        self.tem_embed = embed(
            self.n_joints, self.n_frames, 64 * 4, norm=False, bias=bias
        )
        self.spa_embed = embed(self.n_joints, self.n_joints, 64, norm=False, bias=bias)
        self.joint_embed = embed(self.n_joints, in_channels, 64, norm=True, bias=bias)
        self.dif_embed = embed(self.n_joints, in_channels, 64, norm=True, bias=bias)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)

        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)

        self.n_out_features = self.dim1 * 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        """
        Args: 
            input (torch.Tensor): Input tensor of shape :math:`(N, in\_channels, T_{in}, V_{in})`
        
        Returns:
            torch.Tensor: Output embedding of shape :math:`(N, n\_out\_features)`

        where
            - :math:`N` is a batch size,
            - :math:`T_{in}` is a length of input sequence,
            - :math:`V_{in}` is the number of graph nodes,
            - :math:`n\_out\_features` is the output embedding dimension.

        """

        # B, C, T, V
        input = input.permute(0, 2, 3, 1)

        bs, step, num_joints, dim = input.size()
        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(input)

        tem1 = self.tem_embed(self.tem_oh).repeat(bs, 1, 1, 1)
        spa1 = self.spa_embed(self.spa_oh).repeat(bs, 1, 1, 1)

        dif = self.dif_embed(dif)
        dy = pos + dif

        # Joint-level Module
        input = torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)

        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)

        # Classification head
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        return output

    def one_hot(self, bs, spa, tem):
        """
        get one-hot encodings
        """
        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)
        return y_onehot
