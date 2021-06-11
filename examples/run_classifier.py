import hydra
from dataclasses import dataclass
import pytorch_lightning as pl
from slr.core.classification_model import ClassificationModel

@hydra.main(config_path="./", config_name="sample_conf")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.fit()

if __name__ == "__main__":
    main()