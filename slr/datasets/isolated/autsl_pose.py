import torch
import pandas as pd
import pickle
import os
import numpy as np
import torch.nn.functional as F


class AUTSLPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file,
        root_dir,
        class_mappings_file_path,
        splits=["train"],
        transforms=None,
        use_scores=True,
    ):

        self.class_mappings_file_path = class_mappings_file_path
        self.data = []
        self.glosses = []
        self.root_dir = root_dir
        self.read_index_file(split_file, splits)
        self.transforms = transforms
        self.use_scores = use_scores

    def read_index_file(self, index_file_path, splits):

        class_mappings_df = pd.read_csv(self.class_mappings_file_path)
        self.id_to_glosses = dict(
            zip(class_mappings_df["ClassId"], class_mappings_df["TR"])
        )
        self.glosses = sorted(self.id_to_glosses.values())

        df = pd.read_csv(index_file_path, header=None)

        modality = "color"
        for i in range(len(df)):
            instance_entry = df[0][i] + "_" + modality + ".mp4", df[1][i]
            self.data.append(instance_entry)

    def load_pose_from_path(self, path):
        """
        Load dumped pose keypoints
        """
        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def read_data(self, index):
        video_name, label = self.data[index]
        pose_path = os.path.join(self.root_dir, video_name.replace(".mp4", ".pkl"))
        pose_data = self.load_pose_from_path(pose_path)
        pose_data["label"] = torch.tensor(label, dtype=torch.long)
        return pose_data

    def __len__(self):
        return len(self.data)

    @property
    def num_class(self):
        return len(self.glosses)

    def __getitem__(self, index):
        """
        Returns
        C - num channels
        T - num frames
        V - num vertices
        M - num persons
        """
        data = self.read_data(index)
        # imgs shape: (T, V, C)
        kps = data["keypoints"]
        scores = data["confidences"]

        if self.use_scores:
            kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

        # Expand to 4 dim for person dim
        if kps.ndim == 3:
            kps = np.expand_dims(kps, axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "pose_kps": torch.tensor(kps).permute(2, 0, 1, 3),  # (C, T, V, M )
            "vid_shape": data["vid_shape"],
            "label": data["label"],
        }

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    @staticmethod
    def collate_fn(batch_list):
        max_frames = max([i["pose_kps"].shape[0] for i in batch_list])
        kps = [
            F.pad(
                x["pose_kps"],
                (0, 0, 0, 0, 0, 0, 0, max_frames - x["pose_kps"].shape[0]),
            )
            for i, x in enumerate(batch_list)
        ]
        kps = torch.stack(kps, dim=0)
        labels = [x["label"] for i, x in enumerate(batch_list)]
        labels = torch.stack(labels, dim=0)

        return dict(frames=kps, labels=labels)