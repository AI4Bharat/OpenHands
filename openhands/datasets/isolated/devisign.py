import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class DeviSignDataset(BaseIsolatedDataset):
    """
    Chinese Isolated Sign language dataset from the paper:
    
    `The devisign large vocabulary of chinese sign language database and baseline evaluations`
    """
    def read_glosses(self):
        self.glosses = []
        df = pd.read_csv(self.split_file, delimiter="\t", encoding="utf-8")
        for i in range(len(df)):
            self.glosses.append(df["Meaning (Chinese)"][i].strip())

        # self.glosses = list(range(2000))
        
        # label_encoder = LabelEncoder()
        # label_encoder.fit(self.glosses)
        # TODO: There seems to be file-encoding issues, hence total glosses don't match with actual
        # print(len(label_encoder.classes_))
        # exit()

    def read_original_dataset(self):
        """
        Check the file "DEVISIGN Technical Report.pdf" inside `Documents\` folder
        for dataset format (page 12) and splits (page 15)
        """

        if "rgb" in self.modality:
            common_filename = "color.avi"
        elif "pose" in self.modality:
            common_filename = "pose.pkl"
        else:
            raise NotImplementedError

        video_files_path = os.path.join(self.root_dir, "**", common_filename)
        video_files = glob(video_files_path, recursive=True)
        if not video_files:
            exit(f"No videos files found for: {video_files_path}")

        for video_file in video_files:
            naming_parts = video_file.replace("\\", "/").split("/")[-2].split("_")
            gloss_id = int(naming_parts[1])
            signer_id = int(naming_parts[0].replace("P", ""))

            if (signer_id <= 4 and "train" in self.splits) or (
                signer_id > 4 and "test" in self.splits
            ):
                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
