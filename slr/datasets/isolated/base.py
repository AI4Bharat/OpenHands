import os
import torch
import torch.nn.functional as F
from .utils import get_data_transforms
from .data_readers import load_pose_from_path


class BaseIsolatedDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.data = []
        self.glosses = []
        VALID_MODALITIES = ["rgb", "rgbd", "pose"]

        self.root_dir = config.get("root_dir")
        self.split_file = config.get("split_file")
        self.class_mappings_file_path = config.get("class_mappings_file_path", None)
        self.splits = config.get("splits", ["train"])
        self.modality = config.get("modality", "rgb")
        self.transforms = config.get("transforms", None)
        self.img_resize_dims = config.get("img_resize_dims", (264, 264))
        self.pose_use_confidence_scores = config.get(
            "pose_use_confidence_scores", False
        )
        self.pose_use_z_axis = config.get("pose_use_z_axis", False)

        self.in_channels = None

        if self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"Expected variable modality to have values [rgb, rgbd, pose]. Obtained {self.modality}"
            )

        self.read_index_file()

        if self.transforms is None:
            self.transforms = get_data_transforms(self.modality, self.img_resize_dims)

    @property
    def num_class(self):
        return len(self.glosses)

    def read_index_file(self):
        """
        Implement this method to read (video_name/video_folder, classification_label)
        into self.data[]
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch_list):
        max_frames = max([x["frames"].shape[1] for x in batch_list])

        frames = [
            F.pad(x["frames"], (0, 0, 0, 0, 0, max_frames - x["frames"].shape[1], 0, 0))
            for i, x in enumerate(batch_list)
        ]

        frames = torch.stack(frames, dim=0)
        labels = [x["label"] for i, x in enumerate(batch_list)]
        labels = torch.stack(labels, dim=0)

        return dict(frames=frames, labels=labels)

    def __getitem__(self, index):
        raise NotImplementedError
