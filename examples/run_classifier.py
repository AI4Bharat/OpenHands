import hydra
from dataclasses import dataclass
import pytorch_lightning as pl
from slr.core.classification_model import ClassificationModel
from slr.core.exp_utils import experiment_manager


@hydra.main(config_path="./", config_name="sample_conf")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    experiment_manager(trainer, cfg.get("exp_manager", None))
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.init_from_checkpoint_if_available()
    model.fit()


if __name__ == "__main__":
    main()