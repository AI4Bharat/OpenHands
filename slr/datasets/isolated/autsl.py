import os
import pandas as pd
from .video_isolated_dataset import VideoIsolatedDataset
from .data_readers import load_frames_from_video


class AUTSLDataset(VideoIsolatedDataset):
    def read_index_file(self):

        class_mappings_df = pd.read_csv(self.class_mappings_file_path)
        self.id_to_glosses = dict(
            zip(class_mappings_df["ClassId"], class_mappings_df["TR"])
        )
        self.glosses = sorted(self.id_to_glosses.values())

        df = pd.read_csv(self.split_file, header=None)

        common_filename = "pose.pkl" if "pose" in self.modality else "color.mp4"
        for i in range(len(df)):
            instance_entry = df[0][i] + "_" + common_filename, df[1][i]
            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
