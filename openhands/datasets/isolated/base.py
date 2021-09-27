import torch
import torch.nn.functional as F
import torchvision
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os, sys
from glob import glob
from natsort import natsorted
from ..video_transforms import *


class BaseIsolatedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file,
        root_dir,
        class_mappings_file_path=None,
        splits=["train"],
        modality="rgb",
        transforms="default",
        cv_resize_dims=(264, 264),
        pose_use_confidence_scores=False,
        pose_use_z_axis=False,
    ):
        super().__init__()

        self.data = []
        self.glosses = []
        self.split_file = split_file
        self.root_dir = root_dir
        self.class_mappings_file_path = class_mappings_file_path
        self.splits = splits
        self.modality = modality
        self.read_index_file()

        self.cv_resize_dims = cv_resize_dims
        self.pose_use_confidence_scores = pose_use_confidence_scores
        self.pose_use_z_axis = pose_use_z_axis

        if "rgb" in modality:
            self.in_channels = 3
            if modality == "rgbd":
                self.in_channels += 1

            self.__getitem = self.__getitem_video

        elif modality == "pose":
            self.in_channels = 4
            if not self.pose_use_confidence_scores:
                self.in_channels -= 1
            if not self.pose_use_z_axis:
                self.in_channels -= 1

            self.__getitem = self.__getitem_pose

        else:
            exit(f"ERROR: Modality `{modality}` not supported")

        self.setup_transforms(modality, transforms)

    def setup_transforms(self, modality, transforms):
        if "rgb" in modality:
            if transforms == "default":
                albumentation_transforms = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                        ),
                        A.ChannelDropout(p=0.1),
                        A.RandomRain(p=0.1),
                        A.GridDistortion(p=0.3),
                    ]
                )
                self.transforms = torchvision.transforms.Compose(
                    [
                        Albumentations2DTo3D(albumentation_transforms),
                        NumpyToTensor(),
                        RandomTemporalSubsample(16),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomCrop(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        TCHW2CTHW(),
                    ]
                )
            elif transforms:
                self.transforms = transforms
            else:
                self.transforms = torchvision.transforms.Compose(
                    [
                        NumpyToTensor(),
                        # THWC2CTHW(),
                        THWC2TCHW(),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        TCHW2CTHW(),
                    ]
                )
        elif "pose" in modality:
            if transforms == "default":
                transforms = None
            self.transforms = transforms

    @property
    def num_class(self):
        return len(self.glosses)

    def read_index_file(self, splits):
        """
        Implement this method to read (video_name/video_folder, classification_label)
        into self.data[]
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def load_pose_from_path(self, path):
        """
        Load dumped pose keypoints.
        Should contain: {
            "keypoints" of shape (T, V, C),
            "confidences" of shape (T, V),
            "vid_shape" of shape (W, H)
        }
        """
        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def read_video_data(self, index):
        raise NotImplementedError
        # return imgs, label, video_id

    def __getitem_video(self, index):
        imgs, label, video_id = self.read_video_data(index)
        # imgs shape: (T, H, W, C)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return {"frames": imgs, "label": torch.tensor(label, dtype=torch.long)}

    @staticmethod
    def collate_fn(batch_list):
        max_frames = max([x["frames"].shape[1] for x in batch_list])
        
        # Pad the temporal dimension to `max_frames` for all videos
        frames = [
            F.pad(x["frames"], (0, 0, 0, max_frames - x["frames"].shape[1], 0, 0))
            for i, x in enumerate(batch_list)
        ]

        frames = torch.stack(frames, dim=0)
        labels = [x["label"] for i, x in enumerate(batch_list)]
        labels = torch.stack(labels, dim=0)

        return dict(frames=frames, labels=labels)

    def read_pose_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        # If `video_path` is folder of frames from which pose was dumped, keep it as it is.
        # Otherwise, just remove the video extension
        pose_path = (
            video_path if os.path.isdir(video_path) else os.path.splitext(video_path)[0]
        )
        pose_path = pose_path + ".pkl"
        pose_data = self.load_pose_from_path(pose_path)
        pose_data["label"] = torch.tensor(label, dtype=torch.long)
        return pose_data, pose_path

    def __getitem_pose(self, index):
        """
        Returns
        C - num channels
        T - num frames
        V - num vertices
        """
        data, path = self.read_pose_data(index)
        # imgs shape: (T, V, C)
        kps = data["keypoints"]
        scores = data["confidences"]

        if not self.pose_use_z_axis:
            kps = kps[:, :, :2]

        if self.pose_use_confidence_scores:
            kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "frames": torch.tensor(kps).permute(2, 0, 1),  # (C, T, V)
            "label": data["label"],
        }

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __getitem__(self, index):
        return self.__getitem(index)
