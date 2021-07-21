import hydra
from slr.core.pretraining_model import PosePretrainingModel

@hydra.main(config_path="./", config_name="pretrain_bert")
def main(cfg):
    trainer = PosePretrainingModel(model_cfg=cfg.model_cfg, params=cfg.params)
    trainer.fit()

if __name__ == "__main__":
    main()