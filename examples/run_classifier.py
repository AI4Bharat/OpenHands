'''
python run_classifier.py --config-path=examples/configs/include/ --config-name=gcn_bert hydra.run.dir=./slr_experiments/INCLUDE50/GCN-BERT/
'''

import hydra
import pytorch_lightning as pl
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer


@hydra.main(config_path="./", config_name="sample_conf")
def main(cfg):
    trainer = get_trainer(cfg)
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.init_from_checkpoint_if_available()
    model.fit()


if __name__ == "__main__":
    main()
