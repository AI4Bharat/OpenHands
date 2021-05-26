import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from .utils import Normalize3D

class BaseVideoIsolatedDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, root_dir, resize_dims=(264, 264), splits=["train"]):
        self.data = []
        self.glosses = []
        self.read_index_file(split_file, splits)
        self.root_dir = root_dir
        self.resize_dims = resize_dims
        self.transforms = torchvision.transforms.Compose(
            [
                # Normalize3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    def num_class(self):
        return len(self.glosses)
    
    def read_index_file(self, splits):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def load_frames_from_video(self, video_path, start_frame, end_frame):
        """
        Load the frames of the video between start and end frames.
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
        print(imgs.shape)
        imgs = torch.tensor(imgs/255.0).permute(3, 0, 1, 2)

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
    