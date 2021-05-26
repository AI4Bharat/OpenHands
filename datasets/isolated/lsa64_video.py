import os
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder
from .base_video import BaseVideoIsolatedDataset


class LSA64Dataset(BaseVideoIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):
        '''
        Dataset includes 3200 videos where 10 non-expert subjects executed 5 repetitions of 64 different types of signs.

        For train-set, we use signers 1-8.
        Val-set & Test-set: Signer-9 & Signer-10
        '''
        df = pd.read_csv(index_file_path, delimiter='|', header=None)

        self.glosses = sorted([df[1][i].strip() for i in range(len(df))])
        self.sign_id2name =  {df[0][i]: df[1][i] for i in range(len(df))}
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        video_files = glob(f"{self.root_dir}/*.mp4")
        for video_file in video_files:
            video_name = os.path.basename(video_file).replace(".mp4", '')
            sign_id, signer_id, repeat_id = map(int, video_name.split('_'))
            
            if (signer_id < 9 and "train" in splits) or (signer_id == 9 and "val" in splits) or (signer_id == 10 and "test" in splits):
                sign_name = self.sign_id2name[sign_id]
                gloss_cat = label_encoder.transform([sign_name])[0]
                instance_entry = video_file, gloss_cat
                self.data.append(instance_entry)
        return

    def read_data(self, index):
        video_path, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_path)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label
