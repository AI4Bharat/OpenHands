import hydra
from openhands.core.wav2vec2 import PosePretrainingModel

@hydra.main(config_path="./", config_name="pretrain_wav2vec2")
def main(cfg):
    trainer = PosePretrainingModel(model_cfg=cfg.model, params=cfg.params)
    trainer.fit()

if __name__ == "__main__":
    main()
