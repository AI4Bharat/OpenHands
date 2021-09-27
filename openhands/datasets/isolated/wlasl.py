import os
import json
from sklearn.preprocessing import LabelEncoder
from .base import BaseIsolatedDataset
from .data_readers import load_frames_from_video

class WLASLDataset(BaseIsolatedDataset):
    def read_index_file(self):
        with open(self.split_file, "r") as f:
            content = json.load(f)

        self.glosses = sorted([gloss_entry["gloss"] for gloss_entry in content])
        label_encoder = LabelEncoder()
        label_encoder.fit(self.glosses)

        for gloss_entry in content:
            gloss, instances = gloss_entry["gloss"], gloss_entry["instances"]
            gloss_cat = label_encoder.transform([gloss])[0]

            for instance in instances:
                if instance["split"] not in self.splits:
                    continue

                video_id = instance["video_id"]
                instance_entry = video_id, gloss_cat
                self.data.append(instance_entry)

        if not self.data:
            exit(f"ERROR: No {splits} data found")

    def read_video_data(self, index):
        video_name, label, start_frame, end_frame = self.data[index]
        video_path = os.path.join(self.root_dir, video_name + ".mp4")
        imgs = load_frames_from_video(video_path, start_frame, end_frame)
        return imgs, label, video_name
