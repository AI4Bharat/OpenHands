import torch
import torch.nn as nn
import torch.functional as F
from .st_gcn import STModel

# Adopted from: https://github.com/TengdaHan/DPC


class DPC_RNN_Pretrainer(nn.Module):
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

