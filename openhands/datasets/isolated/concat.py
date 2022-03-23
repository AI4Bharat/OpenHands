import os

from .base import BaseIsolatedDataset
from .asllvd import ASLLVDDataset
from .autsl import AUTSLDataset
from .bosphorus22k import Bosphorus22kDataset
from .csl import CSLDataset
from .devisign import DeviSignDataset
from .gsl import GSLDataset
from .include import INCLUDEDataset
from .lsa64 import LSA64Dataset
from .msasl import MSASLDataset
from .rwth_phoenix_weather_signer03_cutout import RWTH_Phoenix_Signer03_Dataset
from .wlasl import WLASLDataset

class ConcatDataset(BaseIsolatedDataset):
    def __init__(self, datasets, unify_vocabulary=False, **kwargs):

        self.unify_vocabulary = unify_vocabulary
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
        self.glosses = set()
        for dataset in self.datasets:
            for class_name in dataset.glosses:
                if self.unify_vocabulary:
                    self.glosses.add(dataset.normalized_class_mappings[class_name])
                else:
                    self.glosses.add(f"{dataset.lang_code}__{class_name}")
        
        # TODO: Make the sequence agnostic to the order in which datasets are listed
        self.glosses = sorted(self.glosses)
    
    def read_original_dataset(self):
        for dataset in self.datasets:
            if dataset.only_metadata:
                continue
            
            for video_name, class_id in dataset.data:
                class_name = dataset.id_to_gloss[class_id]
                if self.unify_vocabulary:
                    class_name = dataset.normalized_class_mappings[class_name]
                else:
                    class_name = f"{dataset.lang_code}__{class_name}"
                
                instance_entry = os.path.join(dataset.root_dir, video_name), self.gloss_to_id[class_name], dataset.lang_code, dataset.__class__.__name__
                self.data.append(instance_entry)
