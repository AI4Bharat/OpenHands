import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import os
import h5py
import pickle
import numpy as np

from ...core.data import create_pose_transforms

class WindowedDatasetHDF5(torch.utils.data.DataLoader):
    """
    Windowed dataset loader from HDF5 for SL-DPC model.

    Args:
        root_dir (str): Directory which contains the data.
        file_format (str): File type. Default: ``h5``.
        transforms (obj | None): Compose object with transforms or None. Default: ``None``.
        seq_len (int): Sequence length for each window. Default: 10. 
        num_seq (int): Total number of windows. Default: 7.
        downsample (int): Number of frames to skip per timestep when sampling. Default: 3.
        num_channels (int): Number of input channels. Default: 2.
    """
    def __init__(
        self,
        root_dir,
        file_format='h5',
        transforms=None,
        seq_len=10,
        num_seq=7,
        downsample=3,
        num_channels=2,
    ):

        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        self.transforms = create_pose_transforms(transforms)

        self.hdf5_files = []
        self.data_list = []

        h5_files = glob.glob(os.path.join(root_dir, "**", f"*.{file_format}"), recursive=True)

        for h5_index, h5_file in enumerate(h5_files):
            hf = h5py.File(h5_file, "r")
            self.hdf5_files.append(hf)

            for data_name in list(hf["keypoints"].keys()):
                self.data_list.append((h5_index, data_name))

    def __len__(self):
        return len(self.data_list)

    def load_pose_from_h5(self, idx):
        h5_index, data_name = self.data_list[idx]
        return self.hdf5_files[h5_index]["keypoints"][data_name]

    def idx_sampler(self, vlen):
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return None
        n = 1
        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n
        )
        seq_idx = (
            np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len
            + start_idx
        )
        seq_idx_block = (
            seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        )
        return [seq_idx_block]

    def __getitem__(self, idx):
        kps = self.load_pose_from_h5(idx)
        seq_len = kps.shape[0]
        items = self.idx_sampler(seq_len)
        if items is None:
            return None

        data_list = []
        for item in items[0]:
            curr_data = kps[item, ...]
            curr_data = curr_data[..., :2]  # Skip z

            curr_data = np.asarray(curr_data, dtype=np.float32)
            data = {
                "frames": torch.tensor(curr_data).permute(2, 0, 1),  # (C, T, V)
            }

            if self.transforms is not None:
                data = self.transforms(data)

            curr_data = data["frames"].permute(1, 2, 0)
            data_list.append(curr_data)

        T, V, C = data_list[0].shape
        t_seq = torch.stack(data_list, 0)
        t_seq = t_seq.view(self.num_seq, T, V, C)

        return t_seq

    def get_weights_for_balanced_sampling(self):
        max_frames = 1*60*60*25 # Assuming 1hr at 25fps
        weights = [0] * len(self.data_list)
        for i in range(len(self.data_list)):
            num_frames = self.load_pose_from_h5(i).shape[0]
            weights[i] = min(num_frames / max_frames, 1.0)
        return torch.DoubleTensor(weights)


class WindowedDatasetPickle(torch.utils.data.DataLoader):
    """
    Windowed dataset loader from HDF5 for SL-DPC model. 
    This module is for loading finetuning datasets.

    Args:
        root_dir (str): Directory which contains the data.
        file_format (str): File type. Default: ``pkl``.
        transforms (obj | None): Compose object with transforms or None. Default: ``None``.
        seq_len (int): Sequence length for each window. Default: 10. 
        num_seq (int): Total number of windows. Default: 10.
        downsample (int): Number of frames to skip per timestep when sampling. Default: 1.
        num_channels (int): Number of input channels. Default: 2.
    """
    def __init__(
        self,
        root_dir,
        file_format='pkl',
        transforms=None,
        seq_len=10,
        num_seq=10,
        downsample=1,
        num_channels=2,
    ):

        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        self.transforms = create_pose_transforms(transforms)

        files_list = glob.glob(os.path.join(root_dir, "**", f"*.{file_format}"), recursive=True)

        self.data_list = []
        for file in files_list:
            data = pickle.load(open(file, "rb"))
            if data["keypoints"].shape[0] > 60:
                self.data_list.append(file)

    def __len__(self):
        return len(self.data_list)

    def load_pose_from_pkl(self, idx):
        file_path = self.data_list[idx]
        pose_data = pickle.load(open(file_path, "rb"))
        kps = pose_data["keypoints"]
        return kps

    def idx_sampler(self, vlen):
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return None
        n = 1
        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n
        )
        seq_idx = (
            np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len
            + start_idx
        )
        seq_idx_block = (
            seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        )
        return [seq_idx_block]

    def __getitem__(self, idx):
        kps = self.load_pose_from_pkl(idx)
        kps = kps[:, :, :2]

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "frames": torch.tensor(kps).permute(2, 0, 1),  # (C, T, V)
        }

        if self.transforms is not None:
            data = self.transforms(data)

        kps = data["frames"].permute(1, 2, 0).numpy()

        seq_len = kps.shape[0]
        items = self.idx_sampler(seq_len)
        if items is None:
            return None

        data_list = []
        for item in items[0]:
            curr_data = kps[item, ...]
            data_list.append(torch.tensor(curr_data))

        T, V, C = data_list[0].shape
        t_seq = torch.stack(data_list, 0)
        t_seq = t_seq.view(self.num_seq, T, V, C)

        return t_seq
