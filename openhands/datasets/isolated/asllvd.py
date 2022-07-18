import os
import pandas as pd
import json

from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_folder

class ASLLVDDataset(BaseIsolatedDataset):
    """
    American Isolated Sign language dataset from the paper:
    
    `The American Sign Language Lexicon Video Dataset <https://ieeexplore.ieee.org/abstract/document/4563181>`
    `The train test split has been taken from the paper <https://arxiv.org/pdf/1901.11164.pdf>`
    """

    lang_code = "ase"
    
    def read_glosses(self):
        glosses = []
        with open(self.class_mappings_file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                glosses.append(line.strip())
        self.glosses = sorted(glosses)

    def read_original_dataset(self):
        with open(self.split_file) as f:
            data = json.load(f)

        for filename in data:
            gloss_cat = self.gloss_to_id[data[filename]['label'].strip('\n\t')]
            instance_entry = filename, gloss_cat
            self.data.append(instance_entry)
