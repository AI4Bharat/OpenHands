import torch
import torch.nn as nn

class PoseFlattener(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_points=27,
        num_persons=1
    ):
        super().__init__()
        self.n_out_features = in_channels*num_points*num_persons

    def forward(self, x):
        """
        x.shape: (B, C, T, V, M)

        Returns:
        out.shape: (T, B, C*V*M)
        """
        x = x.permute(2, 0, 1, 3, 4)
        return torch.flatten(x, start_dim=2)
