import os
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class INCLUDEDataset(BaseIsolatedDataset):
    """
    Indian Isolated Sign language dataset from the paper:
    
    `INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition <https://dl.acm.org/doi/10.1145/3394171.3413528>`_
    """
    def read_glosses(self):
        # TODO: Separate the classes into a separate file?
        df = pd.read_csv(self.split_file)
        self.glosses = sorted({df["Word"][i].strip() for i in range(len(df))})

    def read_original_dataset(self):
        df = pd.read_csv(self.split_file)
        for i in range(len(df)):
            gloss_cat = self.label_encoder.transform([df["Word"][i]])[0]
            instance_entry = df["FilePath"][i], gloss_cat

            video_path = os.path.join(self.root_dir, df["FilePath"][i])
            # TODO: Replace extension with pkl for pose modality?
            if "rgb" in self.modality and not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                continue
            if "/Second (Number)/" in video_path:
                print(f"WARNING: Skipping {video_path} assuming no present")
                continue

            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
