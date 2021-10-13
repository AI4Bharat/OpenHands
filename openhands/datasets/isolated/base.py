import torch
import torch.nn.functional as F
import torchvision
import pickle
import albumentations as A
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os, warnings
from ..video_transforms import *
from ..data_readers import *

class BaseIsolatedDataset(torch.utils.data.Dataset):
    """
    This module provides the datasets for Isolated Sign Language Classification.
    Do not instantiate this class
    """
    def __init__(
        self,
        root_dir,
        split_file=None,
        class_mappings_file_path=None,
        splits=["train"],
        modality="rgb",
        transforms="default",
        cv_resize_dims=(264, 264),
        pose_use_confidence_scores=False,
        pose_use_z_axis=False,
        inference_mode=False,

        # Windowing
        seq_len=1,
        num_seq=1,
    ):
        super().__init__()

        self.split_file = split_file
        self.root_dir = root_dir
        self.class_mappings_file_path = class_mappings_file_path
        self.splits = splits
        self.modality = modality
        self.seq_len = seq_len
        self.num_seq = num_seq
        
        self.glosses = []
        self.read_glosses()
        if not self.glosses:
            raise RuntimeError("Unable to read glosses list")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.glosses)
        print(f"Found {len(self.glosses)} classes in {splits} splits")

        self.data = []
        self.inference_mode = inference_mode
        if inference_mode:
            # Will have null labels
            self.enumerate_data_files(self.root_dir)
        else:
            self.read_original_dataset()
        if not self.data:
            raise RuntimeError("No data found")

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

    def read_glosses(self):
        """
        Implement this method to construct `self.glosses[]`
        """
        raise NotImplementedError
    
    def read_original_dataset(self):
        """
        Implement this method to read (video_name/video_folder, classification_label)
        into self.data[]
        """
        raise NotImplementedError
    
    def enumerate_data_files(self, dir):
        """
        Lists the video files from given directory.
        - If pose modality, generate `.pkl` files for all videos in folder.
          - If no videos present, check if some `.pkl` files already exist
        """
        files = list_all_videos(dir)

        if self.modality == "pose":
            holistic = None
            pose_files = []

            for video_file in files:
                pose_file = os.path.splitext(video_file)[0] + ".pkl"
                if not os.path.isfile(pose_file):
                    # If pose is not cached, generate and store it.
                    if not holistic:
                        # Create MediaPipe instance
                        from ..pipelines.generate_pose import MediaPipePoseGenerator
                        holistic = MediaPipePoseGenerator()
                    # Dump keypoints
                    frames = load_frames_from_video(video_file)
                    holistic.generate_keypoints_for_frames(frames, pose_file)
                    
                pose_files.append(pose_file)
            
            if not pose_files:
                pose_files = list_all_files(dir, extensions=[".pkl"])
            
            files = pose_files
        
        if not files:
            raise RuntimeError(f"No files found in {dir}")
        
        self.data = [(f, -1) for f in files]
        # -1 means invalid label_id

    def __len__(self):
        return len(self.data)

    def load_pose_from_path(self, path):
        """
        Load dumped pose keypoints.
        Should contain: {
            "keypoints" of shape (T, V, C),
            "confidences" of shape (T, V)
        }
        """
        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def read_video_data(self, index):
        """
        Extend this method for dataset-specific formats
        """
        video_path, label = self.data[index]
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name

    def __getitem_video(self, index):
        if self.inference_mode:
            imgs, label, video_id = super().read_video_data(index)
        else:
            imgs, label, video_id = self.read_video_data(index)
        # imgs shape: (T, H, W, C)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return {
            "frames": imgs,
            "label": torch.tensor(label, dtype=torch.long),
            "file": video_id,
        }

    @staticmethod
    def collate_fn(batch_list):
        if "num_windows" in batch_list[0]:
            # Padding not required for windowed models
            frames=[x["frames"] for x in batch_list]
        else:
            max_frames = max([x["frames"].shape[1] for x in batch_list])
            # Pad the temporal dimension to `max_frames` for all videos
            # Assumes each instance of shape: (C, T, V) 
            # TODO: Handle videos (C,T,H,W)
            frames = [
                F.pad(x["frames"], (0, 0, 0, max_frames - x["frames"].shape[1], 0, 0))
                for i, x in enumerate(batch_list)
            ]

        frames = torch.stack(frames, dim=0)
        labels = [x["label"] for i, x in enumerate(batch_list)]
        labels = torch.stack(labels, dim=0)

        return dict(frames=frames, labels=labels, files=[x["file"] for x in batch_list])

    def read_pose_data(self, index):
        if self.inference_mode:
            pose_path, label = self.data[index]
        else:
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
            "file": path,
        }

        if self.transforms is not None:
            data = self.transforms(data)
        
        if self.seq_len > 1 and self.num_seq > 1:
            data["num_windows"] = self.num_seq
            kps = data["frames"].permute(1, 2, 0).numpy() # CTV->TVC
            if kps.shape[0] < self.seq_len * self.num_seq:
                pad_kps = np.zeros(
                    ((self.seq_len * self.num_seq) - kps.shape[0], *kps.shape[1:])
                )
                kps = np.concatenate([pad_kps, kps])

            elif kps.shape[0] > self.seq_len * self.num_seq:
                kps = kps[: self.seq_len * self.num_seq, ...]

            SL = kps.shape[0]
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(torch.tensor(kps[i : i + self.seq_len, ...], dtype=torch.float32))
                i += self.seq_len

            t_seq = torch.stack(clips, 0)
            data["frames"] = t_seq.permute(0, 3, 1, 2) # WTVC->WCTV

        return data

    def __getitem__(self, index):
        return self.__getitem(index)
