import torch
import random


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
        kps = data["pose_kps"]

        kps[0, ...] /= shape[0]
        kps[1, ...] /= shape[1]

        data["pose_kps"] = kps
        return data


class PoseTemporalSubsample:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, data):
        x = data["pose_kps"]
        C, T, V, M = x.shape

        t = x.shape[1]
        if t >= self.num_frames:
            start_index = random.randint(0, t - self.num_frames)
            indices = torch.arange(start_index, start_index + self.num_frames)
            data["pose_kps"] = torch.index_select(x, 1, indices)

        else:
            # Padding
            pad_len = self.num_frames - t
            pad_tensor = torch.zeros(C, pad_len, V, M)
            data["pose_kps"] = torch.cat((x, pad_tensor), dim=1)

        return data


class PoseRandomShift:
    def __call__(self, data):
        x = data["pose_kps"]
        C, T, V, M = x.shape
        data_shifted = torch.zeros_like(x)
        valid_frame = ((x != 0).sum(dim=3).sum(dim=2).sum(dim=0) > 0).long()
        begin = valid_frame.argmax()
        end = len(valid_frame) - torch.flip(valid_frame, dims=[0]).argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shifted[:, bias : bias + size, :, :] = x[:, begin:end, :, :]

        data["pose_kps"] = data_shifted
        return data


class PoseSelect:
    """
    Select the given index keypoints from all keypoints
    """

    def __init__(self, pose_indexes):
        self.pose_indexes = pose_indexes

    def __call__(self, data):

        x = data["pose_kps"]
        x = x[:, :, self.pose_indexes, :]
        data["pose_kps"] = x
        return data
