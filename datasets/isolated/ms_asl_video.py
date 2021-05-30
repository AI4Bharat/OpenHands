import os
import json
from .base_video import BaseVideoIsolatedDataset

class MSASLDataset(BaseVideoIsolatedDataset):
    def read_index_file(self, index_file_path, splits):
        for file in os.listdir(index_file_path):
            path = os.path.join(index_file_path, file)
            instance = json.load(open(path))[0]
            self.data.append(instance)
        
        self.glosses = list(set(sorted(i["text"] for i in self.data)))
        self.id_to_glosses = {i["label"]:i["text"] for i in self.data}
        
    def read_data(self, index):
        instance_data = self.data[index]
        video_id = instance_data["video_id"]
        label = instance_data["label"]
        video_path = os.path.join(self.root_dir, "videos", video_id+".mp4")
        imgs = self.load_frames_from_video(video_path)
        return imgs, label
    