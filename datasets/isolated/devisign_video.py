import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base_video import BaseVideoIsolatedDataset

class DeviSignDataset(BaseVideoIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
        """
        Check the file "DEVISIGN Technical Report.pdf" inside `Documents\` folder
        for dataset format (page 12) and splits (page 15)
        """
        self.glosses = []
        df = pd.read_csv(index_file_path, delimiter='\t')
        for i in range(len(df)):
            self.glosses.append(df["Meaning (Chinese)"][i].strip())
        
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        if "rgb" in modality:
            video_files_path = os.path.join(self.root_dir, "**", "*.avi")
            video_files = glob(video_files_path, recursive=True)
            if not video_files:
                exit(f"No videos files found for: {video_files_path}")
            
            for video_file in video_files:
                naming_parts = video_file.replace('\\', '/').split('/')[-2].split('_')
                gloss_id = int(naming_parts[2])
                signer_id = int(naming_parts[0].replace('P', ''))

                if (signer_id <= 4 and "train" in splits) or (signer_id > 4 and "test" in splits):
                    instance_entry = video_file, gloss_id
                    self.data.append(instance_entry)
        else:
            raise NotImplementedError

    def read_data(self, index):
        video_path, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_path)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label
