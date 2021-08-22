import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .video_isolated_dataset import VideoIsolatedDataset
from .data_readers import load_frames_from_video


class DeviSignDataset(VideoIsolatedDataset):
    def read_index_file(self):
        """
        Check the file "DEVISIGN Technical Report.pdf" inside `Documents` folder
        for dataset format (page 12) and splits (page 15)
        """
        self.glosses = []
        df = pd.read_csv(self.split_file, delimiter="\t", encoding="utf-8")
        for i in range(len(df)):
            self.glosses.append(df["Meaning (Chinese)"][i].strip())

        # TODO: There seems to be file-encoding issues, hence total glosses don't match with actual
        common_filename = "pose.pkl" if "pose" in self.modality else "color.avi"
        video_files_path = os.path.join(self.root_dir, "*", common_filename)
        video_files = glob(video_files_path, recursive=True)
        if not video_files:
            raise ValueError(
                f"Expected variable video_files to be non-empty. {video_files_path} is empty"
            )

        signs = set()
        for video_file in video_files:
            naming_parts = video_file.replace("\\", "/").split("/")[-2].split("_")
            gloss_id = int(naming_parts[1])
            signs.add(gloss_id)
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
