import torch
import random
import numpy as np


class Compose:
    """
    Compose a list of pose transorms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class PoseNormalize:
    """
    Normalize pose keypoints with Width and Height
    """

    def __call__(self, data):
        assert "vid_shape" in data.keys(), "Video shape if needed for normalize"
        shape = data["vid_shape"]
        kps = data["frames"]

        kps[0, ...] /= shape[0]
        kps[1, ...] /= shape[1]

        data["frames"] = kps
        return data


class PoseTemporalSubsample:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, data):
        x = data["frames"]
        C, T, V, M = x.shape

        t = x.shape[1]
        if t >= self.num_frames:
            start_index = random.randint(0, t - self.num_frames)
            indices = torch.arange(start_index, start_index + self.num_frames)
            data["frames"] = torch.index_select(x, 1, indices)

        else:
            # Padding
            pad_len = self.num_frames - t
            pad_tensor = torch.zeros(C, pad_len, V, M)
            data["frames"] = torch.cat((x, pad_tensor), dim=1)

        return data


class PoseRandomShift:
    def __call__(self, data):
        x = data["frames"]
        C, T, V, M = x.shape
        data_shifted = torch.zeros_like(x)
        valid_frame = ((x != 0).sum(dim=3).sum(dim=2).sum(dim=0) > 0).long()
        begin = valid_frame.argmax()
        end = len(valid_frame) - torch.flip(valid_frame, dims=[0]).argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shifted[:, bias : bias + size, :, :] = x[:, begin:end, :, :]

        data["frames"] = data_shifted
        return data


class PoseSelect:
    """
    Select the given index keypoints from all keypoints
    """

    def __init__(self, pose_indexes):
        self.pose_indexes = pose_indexes

    def __call__(self, data):

        x = data["frames"]
        x = x[:, :, self.pose_indexes, :]
        data["frames"] = x
        return data


# Adopted from: https://github.com/AmitMY/pose-format/
class ShearTransform:
    def __init__(self, shear_std=0.2):
        self.shear_std = shear_std

    def __call__(self, data):
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported"
        x = x.permute(1, 3, 2, 0)
        shear_matrix = torch.eye(2)
        shear_matrix[0][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.shear_std, size=1)[0]
        )
        res = torch.matmul(x, shear_matrix)
        data["frames"] = res.permute(3, 0, 2, 1)
        return data


class RotatationTransform:
    def __init__(self, rotation_std=0.2):
        self.rotation_std = rotation_std

    def __call__(self, data):
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported"
        x = x.permute(1, 3, 2, 0)
        rotation_angle = torch.tensor(
            np.random.normal(loc=0, scale=self.rotation_std, size=1)[0]
        )
        rotation_cos = torch.cos(rotation_angle)
        rotation_sin = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor(
            [[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]],
            dtype=torch.float32,
        )
        res = torch.matmul(x, rotation_matrix)
        data["frames"] = res.permute(3, 0, 2, 1)
        return data


class ScaleTransform:
    def __init__(self, scale_std=0.2):
        self.scale_std = scale_std

    def __call__(self, data):
        x = data["frames"]
        assert x.shape[0] == 2, "Only 2 channels inputs supported"

        x = x.permute(1, 3, 2, 0)
        scale_matrix = torch.eye(2)
        scale_matrix[1][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.scale_std, size=1)[0]
        )
        res = torch.matmul(x, scale_matrix)
        data["frames"] = res.permute(3, 0, 2, 1)
        return data
