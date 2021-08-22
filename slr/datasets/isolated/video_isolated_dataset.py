import torch
from .base import BaseIsolatedDataset


class VideoIsolatedDataset(BaseIsolatedDataset):
    def __init__(self, config):
        super().__init__(config)
        self.in_channels = 3 if self.modality == "rgb" else 4

    def read_video_data(self, index):
        # return: imgs, label, video_id
        raise NotImplementedError

    def __getitem__(self, index):
        # imgs shape: (T, H, W, C)
        imgs, label, _ = self.read_video_data(index)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return {"frames": imgs, "label": torch.tensor(label, dtype=torch.long)}
