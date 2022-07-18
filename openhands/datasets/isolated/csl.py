import os
from glob import glob
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class CSLDataset(BaseIsolatedDataset):
    """
    Chinese Isolated Sign language dataset from the paper:
    
    `Attention-Based 3D-CNNs for Large-Vocabulary Sign Language Recognition <https://ieeexplore.ieee.org/document/8466903>`_
    """

    lang_code = "csl"

    def read_glosses(self):
        self.glosses = []
        with open(self.class_mappings_file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.glosses.append(line.strip())

    def read_original_dataset(self):
        """
        Format for word-level CSL dataset:
        1.  naming: P01_25_19_2._color.mp4
            P01: 1, signer ID (person)
            25_19: (25-1)*20+19=499, label ID
            2: 2, the second time performing the sign

        2.  experiment setting:
            split:
                train set: signer ID, [0, 1, ..., 34, 35]
                test set: signer ID, [36, 37, ... ,48, 49]
        """

        if "rgb" in self.modality:
            format = ".mp4"
        elif "pose" in self.modality:
            format = ".pkl"
        else:
            raise ValueError("Unsupported modality: " + self.modality)

        if self.split_file:
            df = pd.read_csv(self.split_file)
            for i in range(len(df)):
                video_path = df["video_path"][i]
                video_file = os.path.join(self.root_dir, video_path)

                if "pose" in self.modality:
                    video_file = video_file.replace(".mp4", format)

                if not os.path.isfile(video_file):
                    raise FileNotFoundError(video_file)

                gloss_id = int(video_file.replace("\\", "/").split("/")[-2])

                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)
        else:
            # Dynamically enumerate from the given directory
            video_files_path = os.path.join(self.root_dir, "**", "*" + format)
            video_files = glob(video_files_path, recursive=True)
            if not video_files:
                exit(f"No videos files found for: {video_files_path}")

            for video_file in video_files:
                gloss_id = int(video_file.replace("\\", "/").split("/")[-2])
                signer_id = int(os.path.basename(video_file).split("_")[0].replace("P", ""))

                if (signer_id <= 35 and "train" in self.splits) or (
                    signer_id > 35 and ("test" in self.splits or "val" in self.splits)
                ):
                    instance_entry = video_file, gloss_id
                    self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
