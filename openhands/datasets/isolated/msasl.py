import os
import json
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class MSASLDataset(BaseIsolatedDataset):
    """
    American Isolated Sign language dataset from the paper:
    
    `MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language <https://arxiv.org/abs/1812.01053>`_
    """

    lang_code = "ase"

    def read_glosses(self):
        self.glosses = set(gloss['text'] for gloss in json.load(open(self.class_mappings_file_path)))

    def read_original_dataset(self):
        for m in json.load(open(self.split_file)):
            filename = m['clean_text'] + '/' + str(m['signer_id'])
            gloss_cat = self.gloss_to_id[m['text']]
            instance = filename, gloss_cat
            self.data.append(instance)

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, "videos", video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
