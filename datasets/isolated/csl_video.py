import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from .base_video import BaseVideoIsolatedDataset

class CSLDataset(BaseVideoIsolatedDataset):
    def read_index_file(self, index_file_path, splits, modality="rgb"):

        self.glosses = []
        with open(index_file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.glosses.append(line.strip())
        if not self.glosses:
            exit(f"ERROR: {index_file_path} is empty")
        
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        if "rgb" in modality:
            video_files_path = os.path.join(self.root_dir, "color", "**", "*.mp4")
            video_files = glob(video_files_path, recursive=True)
            if not video_files:
                exit(f"No videos files found for: {video_files_path}")
            
            for video_file in video_files:
                gloss_id = int(video_file.replace('\\', '/').split('/')[-2])

                instance_entry = video_file, gloss_id
                self.data.append(instance_entry)
        else:
            raise NotImplementedError

    def read_data(self, index):
        video_path, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_path)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label
