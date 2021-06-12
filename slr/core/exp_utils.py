import os
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.base import LoggerCollection


def experiment_manager(trainer, cfg):
    if cfg is None:
        return

    exp_dir = cfg.exp_dir
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    trainer._default_root_dir = exp_dir

    if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
        configure_loggers(
            trainer,
            exp_dir,
            cfg.name,
            None,  # Version
            cfg.create_tensorboard_logger,
            None,  # cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
        )


def configure_loggers(
    trainer,
    exp_dir,
    name,
    version,
    create_tensorboard_logger,
    summary_writer_kwargs,
    create_wandb_logger,
    wandb_kwargs,
):
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}

        tensorboard_logger = TensorBoardLogger(
            save_dir=exp_dir, name=name, version=version, **summary_writer_kwargs
        )
        logger_list.append(tensorboard_logger)

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        wandb_logger = WandbLogger(save_dir=exp_dir, version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)

    logger_list = LoggerCollection(logger_list)

    trainer.logger_connector.configure_logger(logger_list)
