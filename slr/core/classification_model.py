import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from slr.models.loader import get_model
from .data import CommonDataModule


class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.cfg = cfg
        self.datamodule = self.create_datamodule(cfg.data)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.model = self.create_model(cfg.model)
        self.trainer = trainer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["frames"])
        loss = F.cross_entropy(y_hat, batch["labels"])
        acc = self.accuracy_metric(F.softmax(y_hat, dim=-1), batch["labels"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["frames"])
        loss = F.cross_entropy(y_hat, batch["labels"])
        acc = self.accuracy_metric(F.softmax(y_hat, dim=-1), batch["labels"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        return self.get_optimizer(self.cfg.optim)

    def create_datamodule(self, cfg):
        return CommonDataModule(cfg)

    def create_model(self, cfg):
        return get_model(cfg, self.datamodule.dataset)

    def get_optimizer(self, conf):
        optimizer_conf = conf["optimizer"]
        optimizer_name = optimizer_conf.get("name")
        optimizer_params = {}
        if hasattr(optimizer_conf, "params"):
            optimizer_params = optimizer_conf.params

        optimizer = getattr(torch.optim, optimizer_name)(
            params=self.model.parameters(), **optimizer_params
        )

        scheduler_conf = conf["scheduler"]
        scheduler_name = scheduler_conf.get("name")
        scheduler_params = {}
        if hasattr(scheduler_conf, "params"):
            scheduler_params = scheduler_conf.params
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer=optimizer, **scheduler_params
        )

        return [optimizer], [scheduler]

    def fit(self):
        self.trainer.fit(self, self.datamodule)
