import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from functools import partial
from ..datasets.ssl.dpc_dataset import *
from ..models.ssl.dpc_rnn import DPC_RNN_Pretrainer,load_weights_from_pretrained


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (B, NP, B2, NS) = mask.size()  # [B, P, B, N,]
    target = (mask == 1).float()
    target.requires_grad = False
    return target, (B, B2, NS, NP)


def collate_without_none(batch, dataset):
    """
    Utility to collate the batch removing ``None`` values if any. 
    ``None`` values will be replaced by choosing random item from the dataset.
    """
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    if len_batch > len(
        batch
    ):  # source all the required samples from the original dataset at random
        diff = len_batch - len(batch)
        for i in range(diff):
            sample = None
            while sample is None:
                sample = dataset[np.random.randint(0, len(dataset))]
            batch.append(sample)

    return torch.utils.data.dataloader.default_collate(batch)

class PretrainingModelDPC(pl.LightningModule):
    """
    Model for pretraining the SL-DPC architecture.
    Paper: https://arxiv.org/abs/2110.05877

    Args:
        cfg (dict): configuration set.
        create_model_only(bool): If ``True`` only the model object will be initialized and can't be used for running the trainer.
    """
    def __init__(self, cfg, create_model_only=False):
        super().__init__()
        self.cfg = cfg
        self.model = DPC_RNN_Pretrainer(**cfg.model)
        
        if "pretrained" in self.cfg:
            self.model = load_weights_from_pretrained(self.model, self.cfg.pretrained)
        
        if create_model_only:
            return None

        # Train dataset
        if cfg.data.train_dataset.file_format == 'h5':
            self.train_dataset = WindowedDatasetHDF5(**cfg.data.train_dataset)
        else:
            self.train_dataset = WindowedDatasetPickle(**cfg.data.train_dataset)
        
        weights = self.train_dataset.get_weights_for_balanced_sampling()
        self.weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights)
        )

        # Val dataset
        if cfg.data.val_dataset.file_format == 'h5':
            self.valid_dataset = WindowedDatasetHDF5(**cfg.data.val_dataset)
        else:
            self.valid_dataset = WindowedDatasetPickle(**cfg.data.val_dataset)

        params = cfg.params
        self.learning_rate = params.get("lr", 1e-4)
        self.max_epochs = params.get("max_epochs", 1)
        self.batch_size = params.get("batch_size", 2)
        self.num_workers = params.get("num_workers", 0)

        self.output_path = Path.cwd() / params.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)

        self.loss_fn = nn.CrossEntropyLoss()

        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path, every_n_epochs=1,save_last=True,save_top_k=5,monitor="valid_loss",
        )
        self.resume_from_checkpoint = params.get("resume_from_checkpoint",None)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch` and calculates the loss and the accuracy.
        """
        input_seq = batch
        B = input_seq.size(0)
        [score_, mask_] = self.model(input_seq.float())

        target_, (_, B2, NS, NP) = process_output(mask_)
        score_flattened = score_.view(B * NP, B2 * NS)
        target_flattened = target_.view(B * NP, B2 * NS)
        target_flattened = target_flattened.argmax(dim=1).to(score_flattened.device)

        loss = self.loss_fn(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(
            score_flattened, target_flattened, (1, 3, 5)
        )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", top1, on_epoch=True, prog_bar=True)

        return {"loss": loss, "train_acc": top1}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the validation dataloader
        passed in as `batch` and calculates the loss and the accuracy.
        """
        input_seq = batch
        B = input_seq.size(0)
        [score_, mask_] = self.model(input_seq.float())

        target_, (_, B2, NS, NP) = process_output(mask_)
        score_flattened = score_.view(B * NP, B2 * NS)
        target_flattened = target_.view(B * NP, B2 * NS)
        target_flattened = target_flattened.argmax(dim=1).to(score_flattened.device)

        loss = self.loss_fn(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(
            score_flattened, target_flattened, (1, 3, 5)
        )

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", top1, on_epoch=True, prog_bar=True)
        self.log("val_acc_top3", top3, on_epoch=True, prog_bar=True)
        self.log("val_acc_top5", top5, on_epoch=True, prog_bar=True)

        return {"valid_loss": loss, "valid_acc": top1}

    def train_dataloader(self):
        """
        Lightning uses the dataloader returned for train dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.weighted_sampler,
            collate_fn=partial(collate_without_none, dataset=self.train_dataset),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Lightning uses the dataloader returned for valid dataloader.
        """
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=partial(collate_without_none, dataset=self.valid_dataset),
        )

    def configure_optimizers(self):
        """
        Returns the optimizer and the LR scheduler to be used by Lightning.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        }
        return [optimizer], [lr_scheduler]

    def fit(self):
        """
        Method to be called to start the training.
        """
        self.trainer = pl.Trainer(
            gpus=1,
            #precision=16,
            max_epochs=self.max_epochs,
            default_root_dir=self.output_path,
            logger=pl.loggers.WandbLogger(),
            gradient_clip_val=self.hparams.get("gradient_clip_val", 1),
            callbacks=[self.checkpoint_callback],
            resume_from_checkpoint=self.resume_from_checkpoint,
        )
        self.trainer.fit(self)
