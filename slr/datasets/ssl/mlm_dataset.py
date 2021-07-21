import pickle
import numpy as np
import torch
import copy
import random
import math
import glob
import os
from tqdm import tqdm

class PoseMLMDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir, transforms=None, mask_type="random_spans", get_directions=True, deterministic_masks=False,
    ):
        """
        mask_type => {"random", "random_spans", "last"}
        """
        # List all raw files
        self.files_list = glob.glob(os.path.join(root_dir, '**', '*.pkl'), recursive=True)
        self.transforms = transforms
        self.mask_type = mask_type
        self.get_directions = get_directions
        self.DIRECTIONS = ["N", "E", "S", "W", "N"]
        self.direction_encoder = {"N": 0, "E": 1, "S": 2, "W": 3}

        self.deterministic_masks = False
        if deterministic_masks:
            self.files_list = sorted(self.files_list)
            self.precomputed_mask_indices = []
            random.seed(10)
            for i in tqdm(range(self.__len__()), desc="Generating masks"):
                d = self.__getitem__(i)
                self.precomputed_mask_indices.append(d["masked_indices"].bool())
            self.deterministic_masks = True # Set as true *finally*

    def __len__(self):
        return len(self.files_list)

    def load_pose_from_path(self, path):
        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def __getitem__(self, idx):
        file_path = self.files_list[idx]
        data = self.load_pose_from_path(file_path)

        kps = data["keypoints"]
        scores = data["confidences"]
        kps = kps[:, :, :2]  # Skip z

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
            d["direction_labels"] = torch.LongTensor(self.get_direction_labels(orig_kps))

        return d

    def convert_kps_to_features(self, kps, idx):
        '''
        kps.shape: (T, V, C)
        '''
        emb_dim = kps.shape[-2:]
        # Add [CLS]
        kps = torch.cat([torch.ones(1, *emb_dim), kps])

        data = copy.deepcopy(kps) # Masked input data for model

        if self.deterministic_masks:
            masked_indices = self.precomputed_mask_indices[idx]
            data[masked_indices] = torch.zeros(*emb_dim)
            return data, kps, masked_indices
        
        masked_indices = torch.zeros(data.shape[0])

        if self.mask_type == "random":
            for pos in range(1, data.shape[0]):
                if random.random() < 0.2:  # 20%
                    data[pos] = torch.zeros(*emb_dim)
                    masked_indices[pos] = 1

        elif self.mask_type == "random_spans":
            seq_len = data.shape[0]
            mask_ratio = 0.2
            min_rand, max_rand = 0.02, 0.1
            mask_num = math.ceil(seq_len * mask_ratio)

            masked_cnt = 0
            outer_loop_counter = 0
            max_retries = 5
            while masked_cnt < mask_num and outer_loop_counter<max_retries:
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
                    inner_loop_counter+=1
                    
                masked_cnt += rand_mask_len
                data[start_index : start_index + rand_mask_len] = torch.zeros(*emb_dim)
                masked_indices[start_index : start_index + rand_mask_len] = 1
                outer_loop_counter+=1

        elif self.mask_type == "last":
            seq_len = data.shape[0]
            mask_ratio = 0.2
            mask_num = math.ceil(seq_len * mask_ratio)

            data[-mask_num:] = torch.zeros(*emb_dim)
            masked_indices[-mask_num:] = 1
        
        else:
            raise ValueError("Unsupported mask_type: " + self.mask_type)

        return data, kps, masked_indices

    def get_new_directions(self, point):
        # TODO: Generalize to 3D?
        diff_x, diff_y = point
        if diff_x>=0 and diff_y>=0:
            return 0
        elif diff_x<=0 and diff_y>=0:
            return 1
        elif diff_x<=0 and diff_y<=0:
            return 2
        elif diff_x>=0 and diff_y<=0:
            return 3
    
    def get_direction_labels(self, data):
        '''
        Get direction targets as in Motion Transformer paper:
        https://dl.acm.org/doi/10.1145/3444685.3446289
        '''
        num_frames, seq_len, _ = data.shape
        directions = np.full((num_frames, seq_len), -1)
        for frame_idx in range(2, num_frames):
            diff = data[frame_idx] - data[frame_idx-1]
            
            directions[frame_idx] = np.apply_along_axis(self.get_new_directions, 1, diff)

        return directions

    def get_direction_labels_vectorized(self, data):
        '''
        Computes the angle between two subsequent keypoints
        using difference of tan inverse between the 2 vectors and finally
        quantize to get quadrant
        '''
        data = data.numpy()
        num_frames, seq_len, _ = data.shape # (T, V, C)
        directions = np.full((num_frames, seq_len), -1)
        
        # First token is [CLS], and no motion vector for first frame
        for frame_idx in range(2, num_frames):
            ang1 = np.arctan2(data[frame_idx-1][:,1], data[frame_idx-1][:,0])
            ang2 = np.arctan2(data[frame_idx][:,1], data[frame_idx][:,0])
            
            r1 = (ang2 - ang1) * (180.0 / math.pi)
            r1[r1<0]+=360
            # Conversion to int should be enough to discretize it into quadrants
            r1 = r1 / 90       
            directions[frame_idx] = r1.astype(np.int8)
        return directions
