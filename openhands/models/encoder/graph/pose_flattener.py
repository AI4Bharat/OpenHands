import torch
import torch.nn as nn


class PoseFlattener(nn.Module):
    """
    Flattens the pose keypoints across the channel dimension.
    
    Args:
        in_channels (int): Number of channels in the input data.
        num_points (int): Number of spatial joints
          
    """
    def __init__(self, in_channels=3, num_points=27):
        super().__init__()
        self.n_out_features = in_channels * num_points

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape :math:`(N, in_channels, T_{in}, V_{in})`
        
        Returns:
            torch.Tensor: Tensor with channel dimension flattened of shape :math:`(N, T_{in}, in\_channels * V_{in})`
        
        where
            - :math:`N` is a batch size,
            - :math:`T_{in}` is a length of input sequence,
            - :math:`V_{in}` is the number of graph nodes,  

        """
        x = x.permute(0, 2, 1, 3)
        return torch.flatten(x, start_dim=2)
