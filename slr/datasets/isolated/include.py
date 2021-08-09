import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base import BaseIsolatedDataset


class INCLUDEDataset(BaseIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
        # `splits` is not used here as we pass the split-specific CSV directly
        df = pd.read_csv(index_file_path)

        self.glosses = sorted({df["Word"][i].strip() for i in range(len(df))})
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        for i in range(len(df)):
            gloss_cat = label_encoder.transform([df["Word"][i]])[0]
            instance_entry = df["FilePath"][i], gloss_cat

            video_path = os.path.join(self.root_dir, df["FilePath"][i])
            if "rgb" in modality and not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                continue
            if "/Second (Number)/" in video_path:
                print(f"WARNING: Skipping {video_path} assuming no present")
                continue

            self.data.append(instance_entry)
        if not self.data:
            exit("No data found")

    def read_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label, video_name
