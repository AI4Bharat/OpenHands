import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from .data import CommonDataModule
from .pose_data import PoseDataModule

from ..models.loader import get_model
from .pose_data import create_transform

class Inference(pl.LightningModule):
    # TODO: Make this a base class for the trainer class?

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # TODO: Do not load train_pipeline if `inference_mode`
        self.datamodule = self.create_datamodule(cfg.data)
        self.datamodule.setup()
        self.model = self.create_model(cfg.model).to('cpu')
        self.model.eval()
    
    def create_datamodule(self, cfg):
        if cfg.modality == "video":
            return CommonDataModule(cfg)
        elif cfg.modality == "pose":
            return PoseDataModule(cfg)
    
    def create_model(self, cfg):
        return get_model(cfg, self.datamodule.train_dataset.in_channels, self.datamodule.train_dataset.num_class)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt

    def infer_valset(self):
        # TODO: Write output to a csv
        dataloader = self.datamodule.val_dataloader()
        total_time_taken, num_steps = 0.0, 0
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            start_time = time.time()
            y_hat = self.model(batch["frames"])
            class_indices = torch.argmax(y_hat, dim=-1)
            total_time_taken += time.time() - start_time
            num_steps += 1
        
        print(f"Avg time per iteration: {total_time_taken*1000.0/num_steps} ms")
