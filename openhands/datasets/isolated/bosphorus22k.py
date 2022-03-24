import os
import pandas as pd
from .base import BaseIsolatedDataset

class Bosphorus22kDataset(BaseIsolatedDataset):
    """
    Turkish Isolated Sign language dataset(Bosphorus22k) from the paper:
    Link to paper: https://arxiv.org/pdf/2004.01283.pdf
    """

    lang_code = "tsm"

    def read_glosses(self):
        df = pd.read_csv(self.class_mappings_file_path)
        all_glosses = [df.iloc[i][2] for i in range(len(df))]
        self.glosses = sorted(set(all_glosses))

    def read_original_dataset(self):
        """
        Dataset includes 22542 videos where  6 signers executed 4+ repetitions of 744 different types of signs.

        For train-set, we use all signers except signer called user_4. It contains 18,018 videos.
        Test-set: Signer for test set is user_4, total number of videos 4525.
        
        """

        file_format = ".pkl" if "pose" in self.modality else ".mp4"
        df = pd.read_csv(self.class_mappings_file_path)
        
        for i in range(df.shape[0]):
            file_name = self.root_dir+"/"+format((df.iloc[i][1]),"04")+"/"+(df.iloc[i][4])+"_"+format((df.iloc[i][-1]),"03")+file_format
            signer_id = int(df.iloc[i][4].split("_")[-1])
            gloss = df.iloc[i][2]
            gloss_cat = self.gloss_to_id[gloss.strip(' \n\t')]
            if (
                ((signer_id) !=4 and "train" in self.splits)
                or (signer_id == 4 and ("test" in self.splits or "val" in self.splits))
            ):
                instance_entry = file_name, gloss_cat
                self.data.append(instance_entry)
        return
