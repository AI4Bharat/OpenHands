import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from .base import BaseIsolatedDataset

class CSLDataset(BaseIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
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
        with open(index_file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.glosses.append(line.strip())
        if not self.glosses:
            exit(f"ERROR: {index_file_path} is empty")
        
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        if "rgb" in modality:
            format = ".mp4"
        elif "pose" in modality:
            format = ".pkl"
        else:
            raise ValueError("Unsupported modality: "+ modality)
        
        video_files_path = os.path.join(self.root_dir, "**", "*"+format)
        video_files = glob(video_files_path, recursive=True)
        if not video_files:
            exit(f"No videos files found for: {video_files_path}")
        
        for video_file in video_files:
            gloss_id = int(video_file.replace('\\', '/').split('/')[-2])
            signer_id = int(os.path.basename(video_file).split('_')[0].replace('P', ''))

            if (signer_id <= 35 and "train" in splits) or (signer_id > 35 and "test" in splits):
                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)

    def read_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label, video_name
