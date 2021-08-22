import os
import numpy as np
import torch
from .base import BaseIsolatedDataset
from .data_readers import load_pose_from_path


class PoseIsolatedDataset(BaseIsolatedDataset):
    def __init__(self, config):
        super().__init__(config)

        self.in_channels = 4
        if not self.pose_use_confidence_scores:
            self.in_channels -= 1
        if not self.pose_use_z_axis:
            self.in_channels -= 1

    def read_pose_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        pose_path = (
            video_path if os.path.isdir(video_path) else os.path.splitext(video_path)[0]
        )
        pose_path = pose_path + ".pkl"
        pose_data = load_pose_from_path(pose_path)
        pose_data["label"] = torch.tensor(label, dtype=torch.long)
        return pose_data, pose_path

    def __getitem__(self, index):
        """
        Returns
        C - num channels
        T - num frames
        V - num vertices
        M - num persons
        """
        data, _ = self.read_pose_data(index)
        # imgs shape: (T, V, C)
        kps = data["keypoints"]
        scores = data["confidences"]

        if not self.pose_use_z_axis:
            kps = kps[:, :, :2]

        if self.pose_use_confidence_scores:
            kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

        # Expand to 4 dim for person dim
        if kps.ndim == 3:
            kps = np.expand_dims(kps, axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "frames": torch.tensor(kps).permute(2, 0, 1, 3),  # (C, T, V, M )
            "label": data["label"],
        }

        if self.transforms is not None:
            data = self.transforms(data)
        return data