import hydra
from openhands.apis.dpc import PretrainingModelDPC


@hydra.main(config_path="./", config_name="pretrain_dpc")
def main(cfg):
    trainer = PretrainingModelDPC(cfg=cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
