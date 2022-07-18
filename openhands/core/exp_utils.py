import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def get_trainer(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    experiment_manager(trainer, cfg.get("exp_manager", None))
    return trainer

def experiment_manager(trainer, cfg=None):
    """
    Helper to manage the folders and callbacks for the experiments.
    """
    if cfg is None:
        return

    # trainer._default_root_dir = cfg.output_path if cfg.output_path else os.getcwd()
    if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
        configure_loggers(
            trainer,
            cfg.create_tensorboard_logger,
            None,  # cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
        )
    if cfg.create_checkpoint_callback:
        configure_checkpointing(trainer, cfg.checkpoint_callback_params)

    if "early_stopping_callback" in cfg.keys() and cfg.early_stopping_callback:
        configure_early_stopping(trainer, cfg.early_stopping_params)


def configure_loggers(
    trainer,
    create_tensorboard_logger,
    summary_writer_kwargs,
    create_wandb_logger,
    wandb_kwargs,
):
    """
    Creates TensorboardLogger and/or WandBLogger and attach them to trainer.
    """
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}

        tensorboard_logger = TensorBoardLogger(
            save_dir="logs", version=None, **summary_writer_kwargs
        )
        logger_list.append(tensorboard_logger)

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        wandb_logger = WandbLogger(**wandb_kwargs)

        logger_list.append(wandb_logger)

    logger_list = LoggerCollection(logger_list)

    trainer.logger_connector.configure_logger(logger_list)


def configure_checkpointing(trainer, cfg):
    """
    Creates ModelCheckpoint callback and and attach it to the trainer.
    """

    # Remove existing callback if any
    trainer.callbacks = [callback for callback in trainer.callbacks if type(callback) is not ModelCheckpoint]
    
    checkpoint_callback = ModelCheckpoint(**cfg)#, dirpath=trainer._default_root_dir)
    trainer.callbacks.append(checkpoint_callback)


def configure_early_stopping(trainer, cfg):
    """
    Creates EarlyStopping callback and and attach it to the trainer.
    """

    # Remove existing callback if any
    trainer.callbacks = [callback for callback in trainer.callbacks if type(callback) is not EarlyStopping]

    early_stopping_callback = EarlyStopping(**cfg)
    trainer.callbacks.append(early_stopping_callback)
