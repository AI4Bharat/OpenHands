import os
import json
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class MSASLDataset(BaseIsolatedDataset):
    """
    American Isolated Sign language dataset from the paper:
    
    `MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language <https://arxiv.org/abs/1812.01053>`_
    """
    def read_glosses(self):
        # TODO: Separate the classes into a separate file?
        self.read_original_dataset()
        self.glosses = list(set(sorted(i["text"] for i in self.metadata)))
        self.id_to_glosses = {i["label"]: i["text"] for i in self.metadata}
    
    def read_original_dataset(self):
        self.metadata = []
        for file in os.listdir(self.split_file):
            path = os.path.join(self.split_file, file)
            metadatum = json.load(open(path))[0]
            self.metadata.append(metadatum)
            instance = metadatum["video_id"] + ".mp4", metadatum["label"]
            self.data.append(instance)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, "videos", video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
