import torch
import torchvision
import random
import math


class Normalize3D(torchvision.transforms.Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid


def random_temporal_subsample(x, num_samples=8, temporal_dim=1):
    """
    Randomly subsamples num_samples continous frames.
    If the length of num_samples is higher, then the video will be repeated.
    """
    t = x.shape[temporal_dim]
    if t > num_samples:
        start_index = random.randint(0, t - num_samples)
        indices = torch.arange(start_index, start_index + num_samples)
    else:
        indices = torch.arange(t)
        indices = torch.tile(indices, (math.ceil(num_samples / t),))
        indices = indices[:num_samples]

    return torch.index_select(x, temporal_dim, indices)


class RandomTemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self._num_samples = num_samples

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return random_temporal_subsample(x, self._num_samples)
