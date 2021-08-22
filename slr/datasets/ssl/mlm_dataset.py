import pickle
import numpy as np
import torch
import copy
import random
import math
import glob
import os
from tqdm import tqdm
import logging
from ...core.pose_data import create_transform


class TemporalSubsample:
    def __init__(self, num_frames, num_channels=2):
        self.num_frames = num_frames
        self.num_channels = num_channels

    def __call__(self, x):
        t = x.shape[0]
        if t >= self.num_frames:
            start_index = random.randint(0, t - self.num_frames)
            return x[
                start_index : start_index + self.num_frames, :, : self.num_channels
            ]
        else:
            x = x[:, :, : self.num_channels]
            T, V, C = x.shape
            pad_len = self.num_frames - T
            pad_tensor = np.zeros((pad_len, V, C))
            return np.concatenate((x, pad_tensor), axis=0)


class UniformSubsample:
    def __init__(self, num_frames, num_channels=2):
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.temporal_dim = 0

    def __call__(self, x):
        t = x.shape[self.temporal_dim]
        indices = torch.linspace(0, t - 1, self.num_frames)
        indices = torch.clamp(indices, 0, t - 1).long()
        x = torch.index_select(x, self.temporal_dim, indices)
        if x.shape[self.temporal_dim] < self.num_frames:
            T, V, C = x.shape
            pad_len = self.num_frames - T
            pad_tensor = torch.zeros(pad_len, V, C)
            x = torch.cat((x, pad_tensor), dim=self.temporal_dim)
        return x[:, :, : self.num_channels]


class PoseMLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        file_format="h5",
        transforms=None,
        mask_type="random_spans",
        deterministic_masks=False,
        mask_ratio=0.2,
        num_channels=2,
        max_frames=128,
        subsampling_method="random",
        get_directions=False,
    ):
        """
        mask_type => {"random", "random_spans", "last"}
        """
        self.file_format = file_format

        if file_format == "pkl":
            self.load_keypoints = self.load_pose_from_pkl

            self.data_list = glob.glob(
                os.path.join(root_dir, "*", "*.pkl"), recursive=True
            )
            logging.info(f"Found {len(self.data_list)} pkl files in {root_dir}")
        elif file_format == "h5":
            self.load_keypoints = self.load_pose_from_h5

            import h5py

            self.hdf5_files = []
            self.data_list = []

            h5_files = glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True)
            logging.info(f"Found {len(h5_files)} hdf5 files in {root_dir}")

            for h5_index, h5_file in enumerate(h5_files):
                hf = h5py.File(h5_file, "r")
                self.hdf5_files.append(hf)

                for data_name in list(hf["keypoints"].keys()):
                    self.data_list.append((h5_index, data_name))
            logging.info(f"Found {len(self.data_list)} data items")
        else:
            raise ValueError(f"file_format: {file_format} not supported!")

        if subsampling_method == "random":
            self.subsampler = TemporalSubsample(max_frames, num_channels)
        elif subsampling_method == "uniform":
            self.subsampler = UniformSubsample(max_frames, num_channels)
        else:
            raise ValueError(f"Unknown sub-sampling method: {subsampling_method}")

        self.max_seq_len = max_frames + 1  # Including [CLS]
        self.num_channels = num_channels

        self.transforms = create_transform(transforms)
        self.mask_type = mask_type
        self.get_directions = get_directions
        self.DIRECTIONS = ["N", "E", "S", "W", "N"]
        self.direction_encoder = {"N": 0, "E": 1, "S": 2, "W": 3}

        self.mask_ratio = mask_ratio
        self.deterministic_masks = False
        if deterministic_masks:
            assert file_format == "pkl"
            self.data_list = sorted(self.data_list)
            self.precomputed_mask_indices = []
            random.seed(10)
            for i in tqdm(range(self.__len__()), desc="Generating masks"):
                d = self.__getitem__(i)
                self.precomputed_mask_indices.append(d["masked_indices"].bool())
            self.deterministic_masks = True  # Set as true *finally*

    def __len__(self):
        return len(self.data_list)

    def load_pose_from_pkl(self, idx):
        file_path = self.data_list[idx]
        pose_data = pickle.load(open(file_path, "rb"))
        kps = pose_data["keypoints"]
        return torch.tensor(kps)

    def load_pose_from_h5(self, idx):
        h5_index, data_name = self.data_list[idx]
        return self.hdf5_files[h5_index]["keypoints"][data_name]

    def __getitem__(self, idx):

        kps = self.load_keypoints(idx)
        kps = self.subsampler(kps)

        if kps.ndim == 3:
            kps = np.expand_dims(kps, axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "frames": torch.tensor(kps).permute(2, 0, 1, 3),  # (C, T, V, M )
        }

        if self.transforms is not None:
            data = self.transforms(data)

        kps = data["frames"].squeeze(-1).permute(1, 2, 0)
        masked_kps, orig_kps, masked_indices = self.convert_kps_to_features(kps, idx)
        d = {}
        d["masked_kps"] = masked_kps
        d["orig_kps"] = orig_kps
        d["masked_indices"] = masked_indices
        if self.get_directions:
            d["direction_labels"] = torch.LongTensor(
                self.get_direction_labels(orig_kps)
            )

        return d

    def convert_kps_to_features(self, kps, idx):
        """
        kps.shape: (T, V, C)
        """
        emb_dim = kps.shape[-2:]
        data = copy.deepcopy(kps)  # Masked input data for model

        if self.deterministic_masks:
            masked_indices = self.precomputed_mask_indices[idx]
            data[masked_indices] = torch.zeros(*emb_dim)
            return data, kps, masked_indices

        masked_indices = torch.zeros(data.shape[0])

        if self.mask_type == "random":
            for pos in range(1, data.shape[0]):
                if random.random() < self.mask_ratio:
                    data[pos] = torch.zeros(*emb_dim)
                    masked_indices[pos] = 1

        elif self.mask_type == "random_spans":
            seq_len = data.shape[0]
            min_rand, max_rand = self.mask_ratio / 8, self.mask_ratio / 2
            mask_num = math.ceil(seq_len * self.mask_ratio)

            masked_cnt = 0
            outer_loop_counter = 0
            max_retries = 5
            while masked_cnt < mask_num and outer_loop_counter < max_retries:
                rand_mask_len = int(random.uniform(min_rand, max_rand) * seq_len)
                start_index = random.randint(1, seq_len - rand_mask_len)

                inner_loop_counter = 0
                # Resample
                while (
                    any(masked_indices[start_index:rand_mask_len])
                    and masked_cnt + rand_mask_len > mask_num
                ) and inner_loop_counter < max_retries:
                    rand_mask_len = int(random.uniform(min_rand, max_rand) * seq_len)
                    start_index = random.randint(0, seq_len - rand_mask_len)
                    inner_loop_counter += 1

                masked_cnt += rand_mask_len
                data[start_index : start_index + rand_mask_len] = torch.zeros(*emb_dim)
                masked_indices[start_index : start_index + rand_mask_len] = 1
                outer_loop_counter += 1

        elif self.mask_type == "last":
            seq_len = data.shape[0]
            mask_num = math.ceil(seq_len * self.mask_ratio)

            data[-mask_num:] = torch.zeros(*emb_dim)
            masked_indices[-mask_num:] = 1

        else:
            raise ValueError("Unsupported mask_type: " + self.mask_type)

        return data, kps, masked_indices

    def get_new_directions(self, point):
        # TODO: Generalize to 3D?
        diff_x, diff_y = point
        if diff_x >= 0 and diff_y >= 0:
            return 0
        elif diff_x <= 0 and diff_y >= 0:
            return 1
        elif diff_x <= 0 and diff_y <= 0:
            return 2
        elif diff_x >= 0 and diff_y <= 0:
            return 3
        raise RuntimeError(
            "Failed to retrieve direction for: " + str(point)
        )  # NaN cases

    def get_direction_labels(self, data):
        """
        Get direction targets as in Motion Transformer paper:
        https://dl.acm.org/doi/10.1145/3444685.3446289
        """
        num_frames, seq_len, _ = data.shape
        directions = np.full((num_frames, seq_len), -1)
        for frame_idx in range(2, num_frames):
            diff = data[frame_idx] - data[frame_idx - 1]

            directions[frame_idx] = np.apply_along_axis(
                self.get_new_directions, 1, diff
            )

        return directions

    def get_direction_labels_vectorized(self, data):
        """
        Computes the angle between two subsequent keypoints
        using difference of tan inverse between the 2 vectors and finally
        quantize to get quadrant
        """
        data = data.numpy()
        num_frames, seq_len, _ = data.shape  # (T, V, C)
        directions = np.full((num_frames, seq_len), -1)

        # First token is [CLS], and no motion vector for first frame
        for frame_idx in range(2, num_frames):
            ang1 = np.arctan2(data[frame_idx - 1][:, 1], data[frame_idx - 1][:, 0])
            ang2 = np.arctan2(data[frame_idx][:, 1], data[frame_idx][:, 0])

            r1 = (ang2 - ang1) * (180.0 / math.pi)
            r1[r1 < 0] += 360
            # Conversion to int should be enough to discretize it into quadrants
            r1 = r1 / 90
            directions[frame_idx] = r1.astype(np.int8)
        return directions
