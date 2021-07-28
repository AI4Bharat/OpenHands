from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..models.ssl.pretrainer import PreTrainingModel
from ..datasets.ssl.mlm_dataset import PoseMLMDataset


def masked_mse_loss(preds, targets, mask):
    mask = mask.bool()
    targets = targets.reshape(*targets.shape[:2], -1)
    out = (preds[mask] - targets[mask]) ** 2
    return out.mean()


def masked_ce_loss(preds, targets, mask):
    mask = mask.bool()
    targets[mask] = -1
    criterion = nn.NLLLoss(ignore_index=-1)
    preds = preds.reshape(-1, preds.shape[-1])
    targets = targets.reshape(-1)

    loss = criterion(preds, targets)
    return loss


class PosePretrainingModel(pl.LightningModule):
    def __init__(self, model_cfg, params, create_model_only=False):
        super().__init__()

        self.use_direction_loss = params.get("use_direction", False)
        
        if model_cfg.type == 'bert':
            from ..models.encoder.bert import BertModel
            base_encoder = BertModel(params["input_dim"], model_cfg.config)
        else:
            raise NotImplementedError(f"Model-type: {model_cfg.type} not supported")
        
        self.model = PreTrainingModel(
            base_encoder,
            params["input_dim"],
            model_cfg.config,
            self.use_direction_loss,
            params.get("d_out_classes", 0),
        )

        if create_model_only:
            return None
        
        self.params = params

        self.train_dataset = PoseMLMDataset(
            **params.train_dataset,
            get_directions=self.use_direction_loss,
        )
        self.val_dataset = PoseMLMDataset(
            **params.val_dataset,
            get_directions=self.use_direction_loss,
        )

        self.learning_rate = params.get("lr", 2e-4)
        self.max_epochs = params.get("max_epochs", 1)
        self.num_workers = params.get("num_workers", 0)
        self.batch_size = params.get("batch_size", 2)

        self.output_path = Path.cwd() / params.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)

        self.reg_loss_weight = params.get("reg_loss_weight", 1.0)
        self.dir_loss_weight = params.get("dir_loss_weight", 1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mask_preds, direction_preds = self.model(batch["masked_kps"])
        reg_loss = masked_mse_loss(
            mask_preds, batch["orig_kps"], batch["masked_indices"]
        )

        loss = self.reg_loss_weight * reg_loss
        if self.use_direction_loss:
            dir_loss = masked_ce_loss(
                direction_preds, batch["direction_labels"], batch["masked_indices"]
            )
            loss += self.dir_loss_weight * dir_loss

            self.log("train_reg_loss", reg_loss)
            self.log("train_dir_loss", dir_loss)

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        mask_preds, direction_preds = self.model(batch["masked_kps"])
        reg_loss = masked_mse_loss(
            mask_preds, batch["orig_kps"], batch["masked_indices"]
        )

        loss = self.reg_loss_weight * reg_loss
        if self.use_direction_loss:
            dir_loss = masked_ce_loss(
                direction_preds, batch["direction_labels"], batch["masked_indices"]
            )
            loss += self.dir_loss_weight * dir_loss

            self.log("val_reg_loss", reg_loss)
            self.log("val_dir_loss", dir_loss)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0 if self.val_dataset.deterministic_masks else self.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        }
        return [optimizer], [lr_scheduler]

    def fit(self):
        
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            verbose=True,
        )

        self.trainer = pl.Trainer(
            gpus=1,
            # precision=16,
            max_epochs=self.max_epochs,
            default_root_dir=self.output_path,
            # logger=self.logger,
            logger=pl.loggers.WandbLogger(),
            gradient_clip_val=self.hparams.get("gradient_clip_val", 1),
            callbacks=[
                self.checkpoint_callback,
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ],
        )
        self.trainer.fit(self)
