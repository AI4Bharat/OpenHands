import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from .video_isolated_dataset import VideoIsolatedDataset
from .data_readers import load_frames_from_video


class CSLDataset(VideoIsolatedDataset):
    def read_index_file(self):
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
        self.glosses = []
        with open(self.split_file, encoding="utf-8") as f:
            for line in f:
                self.glosses.append(line.strip())
        if not self.glosses:
            raise ValueError(
                f"Expected variable glosses to be non-empty. {self.split_file} is empty"
            )

        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        file_extension = ".pkl" if "pose" in self.modality else ".mp4"
        video_files_path = os.path.join(self.root_dir, "*", "*" + file_extension)
        video_files = glob(video_files_path, recursive=True)
        if not video_files:
            raise ValueError(
                f"Expected variable video_files to be non-empty. {video_files_path} is empty"
            )

        for video_file in video_files:
            gloss_id = int(video_file.replace("\\", "/").split("/")[-2])
            signer_id = int(os.path.basename(video_file).split("_")[0].replace("P", ""))

            if (signer_id <= 35 and "train" in self.splits) or (
                signer_id > 35 and "test" in self.splits
            ):
                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
