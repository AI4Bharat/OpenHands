import os

from .base import BaseIsolatedDataset
from .autsl import AUTSLDataset
from .csl import CSLDataset
from .devisign import DeviSignDataset
from .gsl import GSLDataset
from .include import INCLUDEDataset
from .lsa64 import LSA64Dataset
from .wlasl import WLASLDataset

class ConcatDataset(BaseIsolatedDataset):
    def __init__(self, datasets, **kwargs):

        self.datasets = []
        
        for dataset_cls_name, dataset_kwargs in datasets.items():
            kwargs_copy = dict(kwargs)
            kwargs_copy.update(dataset_kwargs)

            dataset_instance = globals()[dataset_cls_name](**kwargs_copy)
            self.datasets.append(dataset_instance)

        super().__init__(root_dir="", multilingual=True, **kwargs)
        del self.datasets

        assert self.modality == "pose", "Only pose modality is currently supported for this dataset"


    def read_glosses(self):
        self.glosses = []
        for dataset in self.datasets:
            for class_name in dataset.glosses:
                self.glosses.append(f"{dataset.lang_code}__{class_name}")
        
        # Make the sequence agnostic to the order in which datasets are listed
        self.glosses = sorted(self.glosses)
    
    def read_original_dataset(self):
        print("Preparing list of data items... This might take a few minutes to complete")
        # TODO: LabelEncoder seems to be the bottleneck. Just handle using dicts

        for dataset in self.datasets:
            for video_name, class_id in dataset.data:
                class_name = dataset.id_to_gloss[class_id]
                class_name = f"{dataset.lang_code}__{class_name}"
                
                instance_entry = os.path.join(dataset.root_dir, video_name), self.gloss_to_id[class_name], dataset.lang_code
                self.data.append(instance_entry)
