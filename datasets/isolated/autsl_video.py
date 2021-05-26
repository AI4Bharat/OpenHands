import os
import pandas as pd
from .base_video import BaseVideoIsolatedDataset


class AUTSLDataset(BaseVideoIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
        if modality == "rgb":
            modality = "color"

        df = pd.read_csv(index_file_path, header=None)
        for i in range(len(df)):
            instance_entry = df[0][i] + "_" + modality + ".mp4", df[1][i]
            self.data.append(instance_entry)

    def read_data(self, index):
        video_path, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_path)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label
