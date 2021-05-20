import torch
import torch.nn.functional as F
import torchvision

import os
import cv2
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .utils import Normalize


class WLASLVideoDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, root, resize_dims=(264, 264), split=["train"]):
        self.data = []
        self.read_index_file(split_file, split)
        self.root = root
        self.resize_dims = resize_dims
        self.transforms = torchvision.transforms.Compose(
            [
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def load_frames_from_video(self, video_path, start_frame, end_frame):
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
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(img, self.resize_dims)
            frames.append(frame)

        return np.asarray(frames, dtype=np.float32)

    def read_index_file(self, index_file_path, split):
        with open(index_file_path, "r") as f:
            content = json.load(f)

        self.glosses = sorted([gloss_entry["gloss"] for gloss_entry in content])
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        for gloss_entry in content:
            gloss, instances = gloss_entry["gloss"], gloss_entry["instances"]
            gloss_cat = label_encoder.transform([gloss])[0]

            for instance in instances:
                if instance["split"] not in split:
                    continue

                frame_end = instance["frame_end"]
                frame_start = instance["frame_start"]
                video_id = instance["video_id"]

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)

    def __len__(self):
        return len(self.data)

    def num_class(self):
        return len(self.glosses)

    def __getitem__(self, index):
        video_name, label, start_frame, end_frame = self.data[index]
        video_path = os.path.join(self.root, video_name + ".mp4")
        imgs = self.load_frames_from_video(video_path, start_frame, end_frame)

        imgs = torch.tensor(imgs).permute(3, 0, 1, 2)

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
