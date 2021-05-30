import torch
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from .utils import *

class BaseVideoIsolatedDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, root_dir, resize_dims=(264, 264), splits=["train"], transforms="default"):
        self.data = []
        self.glosses = []
        self.root_dir = root_dir
        self.read_index_file(split_file, splits)
        self.resize_dims = resize_dims
        if transforms == "default":
            albumentation_transforms = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.ChannelDropout(p=0.1),
                    A.RandomRain(p=0.1),
                    A.GridDistortion(p=0.3)
                ]
            )
            self.transforms = torchvision.transforms.Compose(
                [
                    Albumentations3D(albumentation_transforms),
                    NumpyToTensor(),
                    THWC2TCHW(),
                    RandomTemporalSubsample(16),
                    torchvision.transforms.RandomCrop((resize_dims[0], resize_dims[1])),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    TCHW2CTHW()
                ]
            )
        else:
            self.transforms = torchvision.transforms.Compose(
                [
                    NumpyToTensor(),
                    THWC2CTHW(),
                ]
            )
    @property
    def num_class(self):
        return len(self.glosses)
    
    def read_index_file(self, splits):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def load_frames_from_video(self, video_path, start_frame, end_frame):
        """
        Load the frames of the video between start and end frames.

        Returns: numpy array of shape (T, H, W, C)
        """
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Temp fix
        if total_frames < start_frame:
            start_frame = 0
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for offset in range(
            min(int(end_frame - start_frame), int(total_frames - start_frame))
        ):
            success, img = vidcap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize_dims)
            frames.append(img)

        return np.asarray(frames, dtype=np.float32)
    
    def load_frames_from_video(self, video_path):
        """
        Load all frames from a video
        """
        frames = []
        
        vidcap = cv2.VideoCapture(video_path)
        while vidcap.isOpened():
            success, img = vidcap.read()
            if not success:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize_dims)
            frames.append(img)

        return np.asarray(frames, dtype=np.float32)
    
    def load_frames_from_folder(self, frames_folder):
        images = natsorted(glob(f"{frames_folder}/*.jpg"))
        if not images:
            exit(f"ERROR: No frames in folder: {frames_folder}")
        
        frames = []
        for img_path in images:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize_dims)
            frames.append(img)

        return np.asarray(frames, dtype=np.float32)

    def read_data(self, index):
        raise NotImplementedError
        # return imgs, label
    
    def __getitem__(self, index):
        imgs, label = self.read_data(index)
        # imgs shape: (T, H, W, C)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return {"frames": imgs, "label": torch.tensor(label, dtype=torch.long)}
    
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
    