import os
import pandas as pd
from .base import BaseIsolatedDataset

class FingerSpellingDataset(BaseIsolatedDataset):
    """
    All fingerspelling dataset ("aed","ase","csl","gsg","gss","ins","tsm")
    """
    dataset_lang_codes = ["ase"]
    

    def read_glosses(self):
        all_glosses=[]
        for lang_code in FingerSpellingDataset.dataset_lang_codes:
            df = pd.read_csv(self.root_dir+"/"+lang_code+"/"+"glosses.csv", header = None)
            lang_glosses = [df.iloc[i][0] for i in range(len(df))]
            all_glosses.extend(lang_glosses)
        self.glosses = sorted(set(all_glosses))
        
    
    def read_original_dataset(self):
        """
        Divided all fingerspelling datasets to 80-20 as test train split
        
        """

        for lang_code in FingerSpellingDataset.dataset_lang_codes:
            df = pd.read_csv(self.root_dir+"/"+lang_code+"/"+"glosses.csv", header = None)
            
            for i in range(df.shape[0]):
                directory = self.root_dir+"/"+lang_code+"/pkl_poses/"+self.splits+"/"+df.iloc[i][0]
                for name in os.listdir(directory):
                    filename = name.split(".")[0]
                    f = directory + "/" + filename
                    gloss = df.iloc[i][0]
                    gloss_cat = self.gloss_to_id[gloss.strip(' \n\t')]
                    instance_entry = f, gloss_cat
                    self.data.append(instance_entry)