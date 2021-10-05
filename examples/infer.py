import hydra
from openhands.core.inference import InferenceModel


@hydra.main(config_path="./", config_name="sample_conf")
def main(cfg):
    model = InferenceModel(cfg=cfg)
    model.init_from_checkpoint_if_available()
    if cfg.data.test_pipeline.dataset.inference_mode:
        model.test_inference()
    else:
        model.compute_test_accuracy()


if __name__ == "__main__":
    main()
