import torch
import random
import math
import numpy as np
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for transform in self.transforms:
            video = transform(video)
        return video

class THWC2TCHW(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class TCHW2CTHW(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2, 3)


class THWC2CTHW(torch.nn.Module):
    def forward(self, x):
        return x.permute(3, 0, 1, 2)


class NumpyToTensor(torch.nn.Module):
    def forward(self, x):
        return torch.from_numpy(x / 255.0)


class Albumentations2DTo3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid):
        """
        Args:
            x (numpy.array): video tensor with shape (T, H, W, C).
        """
        seed = random.randint(0, 99999)
        aug_vid = []
        for x in vid:
            random.seed(seed)
            aug_vid.append((self.transforms(image=np.asarray(x)))["image"])
        return np.stack(aug_vid)


class RandomTemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self._num_samples = num_samples

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): video tensor with shape (T, C, H, W).
        """
        return self.random_temporal_subsample(x, self._num_samples)

    def random_temporal_subsample(self, x, num_samples=8, temporal_dim=0):
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


class PackSlowFastPathway(torch.nn.Module):
    def __init__(self, alpha):
        super().__init()
        self.alpha = alpha

    def forward(self, x):
        return self.pack_pathway(x)

    def pack_pathway(self, frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
