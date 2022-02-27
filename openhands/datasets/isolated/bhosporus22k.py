import os
import pandas as pd
from glob import glob
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class Bhosporus22kDataset(BaseIsolatedDataset):
    """
    Turkis Isolated Sign language dataset(Bhosporus22k) from the paper:
    Link to paper: https://arxiv.org/pdf/2004.01283.pdf
    """
    def read_glosses(self):
        df = pd.read_csv(self.class_mappings_file_path, delimiter=",")
        print(self.class_mappings_file_path)
        all_glosses = [df.iloc[i][2] for i in range(len(df))]
        unique_glosses=set(all_glosses)
        self.glosses = list(unique_glosses)
        #print(self.glosses)

    def read_original_dataset(self):
        """
        Dataset includes 3200 videos where 10 non-expert subjects executed 5 repetitions of 64 different types of signs.

        For train-set, we use signers 1-8.
        Val-set & Test-set: Signer-9 & Signer-10
        
        """

        file_format = ".pkl" if "pose" in self.modality else ".mp4"
        df = pd.read_csv(self.class_mappings_file_path, delimiter=",")
        
        for i in range(df.shape[0]):
            file_name=self.root_dir+"/"+format((df.iloc[i][1]),"04")+"/"+(df.iloc[i][4])+"_"+format((df.iloc[i][-1]),"03")+file_format
            signer_id=df.iloc[i][4].split("_")[-1]
            signer_id=int(signer_id)
            gloss=(df.iloc[i][2])
            gloss_cat = self.label_encoder.transform([gloss.strip(' \n\t')])[0]
            if (
                ((signer_id) !=4 and "train" in self.splits)
                or (signer_id == 4 and "val" in self.splits)
            ):
                instance_entry = file_name, gloss_cat
                #print(instance_entry)
                self.data.append(instance_entry)
        return

