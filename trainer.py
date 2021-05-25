import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class TrainingRunner(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        
        self.model = model
        self.params = params
        self.train_dataset = self.params.get("train_dataset")
        self.valid_dataset = self.params.get("valid_dataset")
        
        self.lr = self.params.get("lr", 2e-4)
        self.max_epochs = self.params.get("max_epochs", 1)
        self.num_workers = self.params.get("num_workers", 0)
        self.batch_size = self.params.get("batch_size", 2)
        
        self.accuracy_metric = pl.metrics.Accuracy()
        
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
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.valid_dataset.collate_fn
        )
    
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        return [optimizer], [scheduler]
    
    def fit(self):  
        self.trainer = pl.Trainer(
#             gpus=1,
#             precision=16,
            max_epochs=self.max_epochs,
        )
        
        self.trainer.fit(self)

if __name__ == "__main__":
    from datasets.isolated.wlasl_video import WLASLVideoDataset
    from models.video_3d import ClassificationModel
    
    dataset = WLASLVideoDataset("WLASL2000/splits/asl2000.json", "WLASL2000/videos")
    model = ClassificationModel(dataset.num_class())