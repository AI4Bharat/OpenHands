import torch
import torch.nn as nn


class PoseFlattener(nn.Module):
    def __init__(self, in_channels=3, num_points=27):
        super().__init__()
        self.n_out_features = in_channels * num_points

    def forward(self, x):
        """
        x.shape: (B, C, T, V)

        Returns:
        out.shape: (B, T, C*V)
        """
        x = x.permute(0, 2, 1, 3)
        return torch.flatten(x, start_dim=2)
