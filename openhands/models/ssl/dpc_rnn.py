import torch
import torch.nn as nn
import torch.nn.functional as F
from .st_gcn import STModel

# Adopted from: https://github.com/TengdaHan/DPC

def load_weights_from_pretrained(model, pretrained_model_path):
    ckpt = torch.load(pretrained_model_path)
    ckpt_dict = ckpt["state_dict"].items()
    pretrained_dict = {k.replace("model.", ""): v for k, v in ckpt_dict}

    model_dict = model.state_dict()
    tmp = {}
    print("\n=======Check Weights Loading======")
    print("Weights not used from pretrained file:")
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print("---------------------------")
    print("Weights not loaded into new model:")
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print("===================================\n")
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    model.to(dtype=torch.float)
    return model


class DPC_RNN_Pretrainer(nn.Module):
    """
    ST-DPC model pretrain module.

    Args:
        pred_steps (int): Number of future prediction steps. Default: 3.
        in_channels (int): Number of channels in the input data. Default: 2.
        hidden_channels (int): Hidden channels for ST-GCN backbone. Default: 64.
        hidden_dim (int): Output dimension from ST-GCN backbone. Default: 256.
        dropout (float): Dropout ratio for ST-GCN backbone. Default: 256.
        graph_args (dict): Parameters for Spatio-temporal graph construction.
        edge_importance_weighting (bool): If ``True``, adds a learnable importance weighting to the edges of the graph. Default: True.
        kwargs (dict): Other parameters for graph convolution units.
        
    """
    def __init__(
        self,
        pred_steps=3,
        in_channels=2,
        hidden_channels=64,
        hidden_dim=256,
        dropout=0.5,
        graph_args={"layout": "mediapipe-27", "strategy": "spatial"},
        edge_importance_weighting=True,
        **kwargs
    ):
        super().__init__()
        
        self.pred_steps = pred_steps

        self.conv_encoder = STModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
            **kwargs
        )

        self.feature_size = hidden_dim
        self.agg = nn.GRU(hidden_dim, self.feature_size, batch_first=True)
        self.network_pred = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_size, self.feature_size),
        )

        self.relu = nn.ReLU(inplace=False)
        self.mask = None
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        """
        Args:
        block (torch.Tensor): Input data of shape :math:`(N, W, T, V, in_channels)`.
        where:
            - :math:`N` is a batch size,
            - :math:`W` is the number of windows,
            - :math:`T` is a length of input sequence,
            - :math:`V` is the number of graph nodes,
            - :math:`in\_channels` is the number of channels.
                
        """
        block = block.permute(0, 1, 4, 2, 3)  # B, N, T, V, C -> B, N, C, T, V
        B, N, C, T, V = block.shape
        block = block.view(B * N, C, T, V)

        feature = self.conv_encoder(block)

        feature_inf_all = feature.view(B, N, self.feature_size)
        feature = self.relu(feature)  # [0, +inf)
        feature = feature.view(B, N, self.feature_size)
        feature_inf = feature_inf_all[:, N - self.pred_steps : :, ...].contiguous()

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0 : N - self.pred_steps, :].contiguous())
        hidden = hidden[-1, :]

        pred = []
        for i in range(self.pred_steps):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden.permute(1, 0, 2)
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_steps, xxx

        N = self.pred_steps
        pred = (
            pred.permute(0, 2, 1)
            .contiguous()
            .view(B * self.pred_steps, self.feature_size)
        )
        feature_inf = (
            feature_inf.permute(0, 1, 2)
            .contiguous()
            .view(B * N, self.feature_size)
            .transpose(0, 1)
        )
        score = torch.matmul(pred, feature_inf).view(B, self.pred_steps, B, N)

        if self.mask is None:
            mask = torch.zeros(
                (B, self.pred_steps, B, N), dtype=torch.int8, requires_grad=False
            ).detach()
            for k in range(B):
                mask[k, :, k, :] = -1  # temporal neg

            tmp = mask.contiguous().view(B, self.pred_steps, B, N)
            for j in range(B):
                tmp[
                    j,
                    torch.arange(self.pred_steps),
                    j,
                    torch.arange(N - self.pred_steps, N),
                ] = 1  # pos
            mask = tmp.view(B, self.pred_steps, B, N)
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)


class DPC_RNN_Finetuner(nn.Module):
    """
    SL-DPC Finetune module.

    This module is proposed in
    `OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages
    <https://arxiv.org/abs/2110.05877>`_
    
    Args:
        num_class (int): Number of classes to classify.
        pred_steps (int): Number of future prediction steps. Default: 3.
        in_channels (int): Number of channels in the input data. Default: 2.
        hidden_channels (int): Hidden channels for ST-GCN backbone. Default: 64.
        hidden_dim (int): Output dimension from ST-GCN backbone. Default: 256.
        dropout (float): Dropout ratio for ST-GCN backbone. Default: 256.
        graph_args (dict): Parameters for Spatio-temporal graph construction.
        edge_importance_weighting (bool): If ``True``, adds a learnable importance weighting to the edges of the graph. Default: True.
        kwargs (dict): Other parameters for graph convolution units.
        
    """
    def __init__(
        self,
        num_class=60,
        pred_steps=2,
        in_channels=2,
        hidden_channels=64,
        hidden_dim=256,
        dropout=0.5,
        graph_args={"layout": "mediapipe-27", "strategy": "spatial"},
        edge_importance_weighting=True,
        **kwargs
    ):
        super().__init__()

        self.pred_steps = pred_steps
        self.num_class = num_class
        self.conv_encoder = STModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
            **kwargs
        )

        self.feature_size = hidden_dim
        self.agg = nn.GRU(hidden_dim, self.feature_size, batch_first=True)

        self.final_bn = nn.BatchNorm1d(self.feature_size)
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feature_size, self.num_class)
        )

        self._initialize_weights(self.final_fc)

    def forward(self, block):
        """
        Args:
        block (torch.Tensor): Input data of shape :math:`(N, W, T, V, in_channels)`.
        where:
            - :math:`N` is a batch size,
            - :math:`W` is the number of windows,
            - :math:`T` is a length of input sequence,
            - :math:`V` is the number of graph nodes,
            - :math:`in\_channels` is the number of channels.
                
        returns:
            torch.Tensor: logits for classification.
        """
        B, N, C, T, V = block.shape
        block = block.view(B * N, C, T, V)

        feature = self.conv_encoder(block)
        feature = F.relu(feature)

        feature = feature.view(B, N, self.feature_size)
        ### aggregate, predict future ###
        context, hidden = self.agg(feature)
        context = context[:, -1, :]
        context = self.final_bn(context)
        output = self.final_fc(context).view(B, self.num_class)

        return output

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
