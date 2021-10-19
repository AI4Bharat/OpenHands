import os
import json
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class WLASLDataset(BaseIsolatedDataset):
    """
    American Isolated Sign language dataset from the paper:
    
    `Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison <https://arxiv.org/abs/1910.11006>`_
    """
    def read_glosses(self):
        with open(self.split_file, "r") as f:
            self.content = json.load(f)
        self.glosses = sorted([gloss_entry["gloss"] for gloss_entry in self.content])

    def read_original_dataset(self):
        for gloss_entry in self.content:
            gloss, instances = gloss_entry["gloss"], gloss_entry["instances"]
            gloss_cat = self.label_encoder.transform([gloss])[0]

            for instance in instances:
                if instance["split"] not in self.splits:
                    continue

                video_id = instance["video_id"]
                instance_entry = video_id, gloss_cat
                self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, label, start_frame, end_frame = self.data[index]
        video_path = os.path.join(self.root_dir, video_name + ".mp4")
        imgs = load_frames_from_video(video_path, start_frame, end_frame)
        return imgs, label, video_name
