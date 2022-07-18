import os
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_folder

class GSLDataset(BaseIsolatedDataset):
    """
    Greek Isolated Sign language dataset from the paper:
    
    `A Comprehensive Study on Deep Learning-based Methods for Sign Language Recognition <https://ieeexplore.ieee.org/document/8466903>`_
    """

    lang_code = "gss"

    def read_glosses(self):
        self.glosses = [
            gloss.strip()
            for gloss in open(self.class_mappings_file_path, encoding="utf-8")
            if gloss.strip()
        ]

    def read_original_dataset(self):
        # CSV Columns: (video_path, gloss_name)
        df = pd.read_csv(self.split_file, delimiter="|", header=None)

        for i in range(len(df)):
            instance_entry = df[0][i], self.gloss_to_id[df[1][i]]
            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = self.load_frames_from_folder(video_path)
        if imgs is None:
            # Some folders don't have images in ".jpg" extension.
            imgs = load_frames_from_folder(video_path, pattern="glosses*")
            if not images:
                exit(f"No images in {video_path}")
        return imgs, label, video_name
