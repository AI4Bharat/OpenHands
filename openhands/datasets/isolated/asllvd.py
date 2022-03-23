import os
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_folder
from bs4 import BeautifulSoup
import json


class ASLLVDDataset(BaseIsolatedDataset):
    """
    American Isolated Sign language dataset from the paper:
    
    `The American Sign Language Lexicon Video Dataset <https://ieeexplore.ieee.org/abstract/document/4563181>`
    `The train test split has been taken from the paper <https://arxiv.org/pdf/1901.11164.pdf>`
    
    """
    
    def read_glosses(self):
        self.glosses = []
        with open(self.class_mappings_file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.glosses.append(line.strip())

    def read_original_dataset(self):
        f = open(self.split_file)
        data = json.load(f)

        for filename in data:
            gloss_cat = self.label_encoder.transform([data[filename]['label'].strip('\n\t')])[0]
            instance_entry = filename, gloss_cat
            self.data.append(instance_entry)
