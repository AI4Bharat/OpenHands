import os
import pandas as pd
from glob import glob
from .base import BaseIsolatedDataset
from ..data_readers import load_frames_from_video

class LSA64Dataset(BaseIsolatedDataset):
    """
    Argentinian Isolated Sign language dataset from the paper:
    
    `LSA64: An Argentinian Sign Language Dataset <http://sedici.unlp.edu.ar/bitstream/handle/10915/56764/Documento_completo.pdf-PDFA.pdf>`_
    """

    lang_code = "aed"

    def read_glosses(self):
        df = pd.read_csv(self.class_mappings_file_path, delimiter="|", header=None)
        self.glosses = [df[1][i].strip() for i in range(len(df))]

    def read_original_dataset(self):
        """
        Dataset includes 3200 videos where 10 non-expert subjects executed 5 repetitions of 64 different types of signs.

        Signer-independent splits:
        For train-set, we use signers 1-8.
        Val-set & Test-set: Signer-9 & Signer-10

        Signer-dependent splits:
        In the original paper, they split randomly, and do not open-source the splits.
        Hence we only follow the signer-based splits we have come-up with (as mentioned above)
        """

        file_format = ".pkl" if "pose" in self.modality else ".mp4"
        video_files = glob(f"{self.root_dir}/*{file_format}")
        for video_file in video_files:
            video_name = os.path.basename(video_file).replace(file_format, "")
            sign_id, signer_id, repeat_id = map(int, video_name.split("_"))
            sign_id -= 1

            if (
                (signer_id < 9 and "train" in self.splits)
                or (signer_id == 9 and "val" in self.splits)
                or (signer_id == 10 and "test" in self.splits)
            ):
                instance_entry = video_file, sign_id
                self.data.append(instance_entry)
        return

    def read_video_data(self, index):
        video_name, label = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name
