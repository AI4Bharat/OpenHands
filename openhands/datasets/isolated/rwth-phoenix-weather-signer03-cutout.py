import os
import pandas as pd
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_folder
from bs4 import BeautifulSoup
# import xlsxwriter


class RWTHPHOENIXSIGNER03Dataset(BaseIsolatedDataset):
    """
    German Isolated Sign language dataset from the paper:
    
    `RWTH-PHOENIX-Weather: A Large Vocabulary Sign Language Recognition and Translation Corpus. <https://www-i6.informatik.rwth-aachen.de/~forster/database-rwth-phoenix.php>`
    Signer03 cutout has been taken for the experiments :
    Image sequence - https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-weather-signer03-cutout-images_20120820.tgz
    Anotations - https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-weather-signer03-cutout_20120820.tgz
    
    """
    def read_glosses(self):
        s = set()
        for path in [self.split_file, self.class_mappings_file_path]:

            with open(path , 'r') as f:
                data = f.read()

                Bs_data = BeautifulSoup(data, "xml")
                glosses=Bs_data.find_all('orth')

                for gloss in glosses:
                    s.add(gloss.text)

        self.glosses = [gloss.strip(' \n\t') for gloss in s]


    def read_original_dataset(self):
        df = pd.read_csv(self.split_file,  header=None)

        with open(self.split_file , 'r') as f:
            data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            filenames = Bs_data.find_all('recording')
            glosses=Bs_data.find_all('orth')

            for filename,gloss in zip(filenames, glosses):
                gloss_cat = self.label_encoder.transform([gloss.text.strip(' \n\t')])[0]
                instance_entry = filename.get('name'), gloss_cat
                self.data.append(instance_entry)
