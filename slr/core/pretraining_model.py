from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..models.ssl.pretrainer import TransformerPreTrainingModel
from ..datasets.ssl.mlm_dataset import PoseMLMDataset
from .pose_data import create_transform

def masked_mse_loss(preds, targets, mask):
    mask = mask.bool()
#     print("targets:", targets.shape)
    targets = targets.reshape(*targets.shape[:2], -1)
    out = (preds[mask]-targets[mask])**2
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
    def __init__(self, model_cfg, params):
        super(PosePretrainingModel, self).__init__()
        self.params = params
        self.model_cfg = model_cfg
        self.train_transforms = create_transform(params.get("train_transforms"))
        self.valid_transforms = create_transform(params.get("valid_transforms"))
        
        self.train_dataset = PoseMLMDataset(params.get("train_data_dir"), self.train_transforms)
        self.val_dataset = PoseMLMDataset(params.get("val_data_dir"), self.valid_transforms, deterministic_masks=True)
        
        self.learning_rate = params.get("lr", 2e-4)
        self.max_epochs = params.get("max_epochs", 1)
        self.num_workers = params.get("num_workers", 0)
        self.batch_size = params.get("batch_size", 2)
        
        self.output_path = Path.cwd() / params.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)
        
        self.model = TransformerPreTrainingModel(params['input_dim'], self.model_cfg, params.get("use_direction", False), params.get("d_out_classes", 0))
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        mask_preds, direction_preds = self.model(batch["masked_kps"])
        reg_loss = masked_mse_loss(mask_preds, batch["orig_kps"], batch["masked_indices"])
        class_loss = masked_ce_loss(direction_preds, batch["direction_labels"], batch["masked_indices"])
            
        loss = class_loss + reg_loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        mask_preds, direction_preds = self.model(batch["masked_kps"])
        reg_loss = masked_mse_loss(mask_preds, batch["orig_kps"], batch["masked_indices"])
        class_loss = masked_ce_loss(direction_preds, batch["direction_labels"], batch["masked_indices"])
            
        loss = class_loss + reg_loss
        self.log("val_loss", loss)
        return {"val_loss": loss}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=0 if self.val_dataset.deterministic_masks else self.num_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        }
        return [optimizer], [lr_scheduler]
        
    def fit(self):
        self.trainer = pl.Trainer(
#             gpus=1,
#             precision=16,
            max_epochs=self.max_epochs,
            default_root_dir=self.output_path,
            logger=self.logger,
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                dirpath=self.output_path,
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
            gradient_clip_val=self.hparams.get("gradient_clip_val", 1),
            num_sanity_val_steps=self.hparams.get("val_sanity_checks", 0),
        )
        self.trainer.fit(self)
