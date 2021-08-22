import os
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .video_isolated_dataset import VideoIsolatedDataset
from .data_readers import load_frames_from_video


class INCLUDEDataset(VideoIsolatedDataset):
    def read_index_file(self):
        # `splits` is not used here as we pass the split-specific CSV directly
        df = pd.read_csv(self.split_file)

        self.glosses = sorted({df["Word"][i].strip() for i in range(len(df))})
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        for i in range(len(df)):
            gloss_cat = label_encoder.transform([df["Word"][i]])[0]
            instance_entry = df["FilePath"][i], gloss_cat

            video_path = os.path.join(self.root_dir, df["FilePath"][i])
            if "rgb" in self.modality and not os.path.isfile(video_path):
                warnings.warn(f"Video file not found: {video_path}")
                continue
            if "/Second (Number)/" in video_path:
                warnings.warn(f"Skipping {video_path}. Assuming not to be present")
                continue

            self.data.append(instance_entry)

        if not self.data:
            raise ValueError(f"Expected variable data to be non-empty")

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
