import os
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class AUTSLDataset(BaseIsolatedDataset):
    """
    Turkish Isolated Sign language dataset from the paper:
    
    `AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset and Baseline Methods <https://arxiv.org/abs/2008.00932>`_
    """

    lang_code = "tsm"

    def read_glosses(self):
        class_mappings_df = pd.read_csv(self.class_mappings_file_path)
        self.id_to_glosses = dict(
            zip(class_mappings_df["ClassId"], class_mappings_df["TR"])
        )
        self.glosses = sorted(self.id_to_glosses.values())
    
    def read_original_dataset(self):
        df = pd.read_csv(self.split_file, header=None)

        if self.modality == "rgb":
            file_suffix = "color.mp4"
        elif self.modality == "pose":
            file_suffix = "color.pkl"

        for i in range(len(df)):
            instance_entry = df[0][i] + "_" + file_suffix, df[1][i]
            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
