import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .video_isolated_dataset import VideoIsolatedDataset
from .data_readers import load_frames_from_folder


class GSLDataset(VideoIsolatedDataset):
    def read_index_file(self):
        self.glosses = [
            gloss.strip()
            for gloss in open(self.class_mappings_file_path, encoding="utf-8")
            if gloss.strip()
        ]
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        # `splits` is not required since pass split-specific CSV directly
        # CSV Columns: (video_path, gloss_name)
        df = pd.read_csv(self.split_file, delimiter="|", header=None)

        for i in range(len(df)):
            gloss_cat = label_encoder.transform([df[1][i]])[0]
            instance_entry = df[0][i], gloss_cat
            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_folder(video_path)
        if imgs is None:
            # Some folders don't have images in ".jpg" extension.
            imgs = load_frames_from_folder(video_path, pattern="glosses*")
            if not imgs:
                raise ValueError(
                    f"Expected variable imgs to be non empty. No images present in {video_path}"
                )
        return imgs, label, video_name
