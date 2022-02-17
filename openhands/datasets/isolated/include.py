import os
import re
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class INCLUDEDataset(BaseIsolatedDataset):
    """
    Indian Isolated Sign language dataset from the paper:
    
    `INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition <https://dl.acm.org/doi/10.1145/3394171.3413528>`_
    """

    lang_code = "ins"

    def read_glosses(self):
        # TODO: Move the classes into a separate file inorder to avoid sorting
        df = pd.read_csv(self.split_file)
        self.glosses = sorted({df["Word"][i] for i in range(len(df))})

        # # Remove serial numbers from gloss names
        # # We are removing it after sorting, because the models we released have classes in the above order
        # self.glosses = [re.sub("\d+\.", '', gloss).strip().upper() for gloss in self.glosses]
        # # Nevermind, this creates issue at `read_original_dataset()`

    def read_original_dataset(self):
        df = pd.read_csv(self.split_file)
        for i in range(len(df)):
            instance_entry = df["FilePath"][i], self.gloss_to_id[df["Word"][i]]

            video_path = os.path.join(self.root_dir, df["FilePath"][i])
            # TODO: Replace extension with pkl for pose modality?
            if "rgb" in self.modality and not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                continue
            if "/Second (Number)/" in video_path:
                print(f"WARNING: Skipping {video_path} assuming not present")
                continue

            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
