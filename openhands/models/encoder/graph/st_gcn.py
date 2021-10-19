import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from .graph_utils import GraphWithPartition


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the graph convolving kernel.
        t_kernel_size (int): Size of the temporal convolving kernel.
        t_stride (int, optional): Stride of the temporal convolution. Default: 1.
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0.
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


class STGCN_BLOCK(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel.
        stride (int, optional): Stride of the temporal convolution. Default: 1.
        dropout (int, optional): Dropout rate of the final output. Default: 0.
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format.
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out},
            V)` format.
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format.
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class STGCN(nn.Module):
    """Spatial temporal graph convolutional network backbone
    
    This module is proposed in
    `Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    <https://arxiv.org/pdf/1801.07455.pdf>`_

    Args:
        in_channels (int): Number of channels in the input data.
        graph_args (dict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable importance weighting to the edges of the graph. Default: True.
        n_out_features (int): Output Embedding dimension. Default: 256. 
        kwargs (dict): Other parameters for graph convolution units.
    """
    def __init__(self, in_channels, graph_args, edge_importance_weighting, n_out_features = 256, **kwargs):
        super().__init__()

        graph_args = OmegaConf.to_container(graph_args)
        self.graph = GraphWithPartition(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.n_out_features = n_out_features
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCN_BLOCK(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
                STGCN_BLOCK(64, 64, kernel_size, 1, **kwargs),
                STGCN_BLOCK(64, 64, kernel_size, 1, **kwargs),
                STGCN_BLOCK(64, 64, kernel_size, 1, **kwargs),
                STGCN_BLOCK(64, 128, kernel_size, 2, **kwargs),
                STGCN_BLOCK(128, 128, kernel_size, 1, **kwargs),
                STGCN_BLOCK(128, 128, kernel_size, 1, **kwargs),
                STGCN_BLOCK(128, 256, kernel_size, 2, **kwargs),
                STGCN_BLOCK(256, 256, kernel_size, 1, **kwargs),
                STGCN_BLOCK(256, self.n_out_features, kernel_size, 1, **kwargs),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        """
        Args: 
            x (torch.Tensor): Input tensor of shape :math:`(N, in\_channels, T_{in}, V_{in})`
        
        Returns:
            torch.Tensor: Output embedding of shape :math:`(N, n\_out\_features)`

        where
            - :math:`N` is a batch size,
            - :math:`T_{in}` is a length of input sequence,
            - :math:`V_{in}` is the number of graph nodes,
            - :math:`n\_out\_features` is the output embedding dimension.

        """
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # NCTV -> NVCT
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous() # NVCT -> NCTV

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        return x
