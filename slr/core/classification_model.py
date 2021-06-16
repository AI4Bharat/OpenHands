import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from slr.models.loader import get_model
from .data import CommonDataModule
from .pose_data import PoseDataModule

class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.cfg = cfg
        self.datamodule = self.create_datamodule(cfg.data)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.model = self.create_model(cfg.model)
        self.trainer = trainer
        self.setup_metrics()

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
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["frames"])
        loss = F.cross_entropy(y_hat, batch["labels"])
        acc = self.accuracy_metric(F.softmax(y_hat, dim=-1), batch["labels"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"valid_loss": loss, "valid_acc": acc}

    def configure_optimizers(self):
        return self.get_optimizer(self.cfg.optim)

    def create_datamodule(self, cfg):
        if cfg.modality == "video":
            return CommonDataModule(cfg)
        elif cfg.modality == "pose":
            return PoseDataModule(cfg)

    def create_model(self, cfg):
        return get_model(cfg, self.datamodule.train_dataset)

    def setup_metrics(self):
        self.accuracy_metric = torchmetrics.functional.accuracy

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

    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        if "resume_from" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["resume_from"]
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt