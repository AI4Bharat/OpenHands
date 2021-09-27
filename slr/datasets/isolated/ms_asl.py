import os
import json
from .base import BaseIsolatedDataset


class MSASLDataset(BaseIsolatedDataset):
    def read_index_file(self, index_file_path, splits):
        self.metadata = []
        for file in os.listdir(index_file_path):
            path = os.path.join(index_file_path, file)
            metadatum = json.load(open(path))[0]
            metadata.append(metadatum)
            instance = metadatum["video_id"] + ".mp4", metadatum["label"]
            self.data.append(instance)

        self.glosses = list(set(sorted(i["text"] for i in metadata)))
        self.id_to_glosses = {i["label"]: i["text"] for i in metadata}

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, "videos", video_name)
        imgs = self.load_frames_from_video(video_path)
        return imgs, label, video_name
