from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..datasets.ssl.mlm_dataset import PoseMLMDataset

from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

class PreTrainingModel(nn.Module):
    def __init__(self, model_config, input_dim, max_seq_len):
        super().__init__()
        # Cfg format: https://huggingface.co/transformers/model_doc/wav2vec2.html#wav2vec2config
        self.model_cfg = Wav2Vec2Config(**model_config)
        self.pretrainer = Wav2Vec2ForPreTraining(self.model_cfg)

        self.w2v2_seq_len = self.pretrainer._get_feat_extract_output_lengths(input_dim*max_seq_len)
        self.n_out_features = self.model_cfg.codevector_dim
    
    def forward_only_transformer(self, input_values):
        outputs = self.pretrainer.wav2vec2(input_values)

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.pretrainer.project_hid(outputs[0])
        return transformer_features
    
    def forward(self, x, get_loss=False):
        '''
        x.shape: (B, T, V, C)
        '''
        if len(x.shape) == 5:
            # Finetuning
            B, C, T, V, M = x.shape
            # Convert to B,T,V*C
            x = x.permute(0, 2, 3, 1, 4).reshape(B,-1)
            # TODO: Generalize the format for all encoders
        else:
            # Pretraining
            B,T,V,C = x.shape
            x = x.reshape(B,-1)
        
        if get_loss:
            # Ref: https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2ForPreTraining
            mask_time_indices = _compute_mask_indices((B, self.w2v2_seq_len), mask_prob=self.model_cfg.mask_time_prob, mask_length=self.model_cfg.mask_time_length, device=self.pretrainer.device)
            outputs = self.pretrainer(x, mask_time_indices=mask_time_indices)
            return outputs, outputs.loss / (B*self.w2v2_seq_len)
        else:
            # self.pretrainer.training = False
            # outputs = self.pretrainer(x)
            # return outputs['projected_states']
            return self.forward_only_transformer(x)

class PosePretrainingModel(pl.LightningModule):
    def __init__(self, model_cfg, params, create_model_only=False):
        super().__init__()
        
        self.params = params
        self.model = PreTrainingModel(model_cfg.config, params['input_dim'], params['max_seq_len'])

        if create_model_only:
            return None

        self.train_dataset = PoseMLMDataset(
            **params.train_dataset,
        )
        # self.val_dataset = PoseMLMDataset(
        #     **params.val_dataset
        # )

        self.learning_rate = params.get("lr", 2e-4)
        self.max_epochs = params.get("max_epochs", 1)
        self.num_workers = params.get("num_workers", 0)
        self.batch_size = params.get("batch_size", 2)

        self.output_path = Path.cwd() / params.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)

    def forward(self, x):
        return self.model(x, get_loss=True)

    def training_step(self, batch, batch_idx):
        outputs, loss = self.model(batch["orig_kps"], get_loss=True)
        
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     mask_preds, direction_preds = self.model(batch["masked_kps"])
    #     reg_loss = masked_mse_loss(
    #         mask_preds, batch["orig_kps"], batch["masked_indices"]
    #     )

    #     loss = outputs.loss
        
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    #     return {"val_loss": loss}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=0 if self.val_dataset.deterministic_masks else self.num_workers,
    #         pin_memory=True,
    #     )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        }
        return [optimizer], [lr_scheduler]

    def fit(self):
        
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            # monitor="val_loss",
            # save_top_k=3,
            # mode="min",
            verbose=True,
        )

        self.trainer = pl.Trainer(
            gpus=1,
            # precision=16,
            max_epochs=self.max_epochs,
            default_root_dir=self.output_path,
            # logger=self.logger,
            # resume_from_checkpoint="/home/gokulnc/SLR/outputs/2021-07-27/05-56-56/model-outputs/epoch=33-step=21011.ckpt",
            logger=pl.loggers.WandbLogger(),
            gradient_clip_val=self.hparams.get("gradient_clip_val", 1),
            callbacks=[
                self.checkpoint_callback,
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ],
        )
        self.trainer.fit(self)
