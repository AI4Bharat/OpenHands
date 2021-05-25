import torch
import torchvision


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
