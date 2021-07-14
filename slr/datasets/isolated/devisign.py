import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base import BaseIsolatedDataset

class DeviSignDataset(BaseIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
        """
        Check the file "DEVISIGN Technical Report.pdf" inside `Documents\` folder
        for dataset format (page 12) and splits (page 15)
        """
        self.glosses = []
        df = pd.read_csv(index_file_path, delimiter='\t', encoding='utf-8')
        for i in range(len(df)):
            self.glosses.append(df["Meaning (Chinese)"][i].strip())
        
        # label_encoder = LabelEncoder()
        # label_encoder.fit(self.glosses)
        # TODO: There seems to be file-encoding issues, hence total glosses don't match with actual
        # print(len(label_encoder.classes_))
        # exit()

        if "rgb" in modality:
            common_filename = "color.avi"
        elif "pose" in modality:
            common_filename = "pose.pkl"
        else:
            raise NotImplementedError
        
        video_files_path = os.path.join(self.root_dir, "**", common_filename)
        video_files = glob(video_files_path, recursive=True)
        if not video_files:
            exit(f"No videos files found for: {video_files_path}")
        
        signs = set()
        for video_file in video_files:
            naming_parts = video_file.replace('\\', '/').split('/')[-2].split('_')
            gloss_id = int(naming_parts[1])
            signs.add(gloss_id)
            signer_id = int(naming_parts[0].replace('P', ''))

            if (signer_id <= 4 and "train" in splits) or (signer_id > 4 and "test" in splits):
                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)
        

    def read_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label, video_name
