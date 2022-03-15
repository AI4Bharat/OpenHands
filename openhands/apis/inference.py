import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from ..core.data import DataModule
from ..models.loader import get_model
from sklearn.metrics import confusion_matrix
import numpy as np

# merge with the corresponding modules in the future release.
class InferenceModel(pl.LightningModule):
    """
    This will be the general interface for running the inference across models.
    Args:
        cfg (dict): configuration set.

    """
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)

        self.model = self.create_model(cfg.model)
        if stage == "test":
            self.model.to('cpu').eval()
    
    def create_model(self, cfg):
        """
        Creates and returns the model object based on the config.
        """
        return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)
    
    def forward(self, x):
        """
        Forward propagates the inputs and returns the model output.
        """
        return self.model(x)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        """
        Intializes the pretrained weights if the ``cfg`` has ``pretrained`` parameter.
        """
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt
    
    def test_inference(self):
        """
        Calculates the time taken for inference for all the batches in the test dataloader.
        """
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        total_time_taken, num_steps = 0.0, 0

        for batch in dataloader:
            start_time = time.time()
            y_hat = self.model(batch["frames"])

            class_indices = torch.argmax(y_hat, dim=-1)
            class_labels = self.datamodule.test_dataset.label_encoder.inverse_transform(class_indices)
            for filename, label in zip(batch["files"], class_labels):
                print(f"{label}:\t{filename}")
            
            total_time_taken += time.time() - start_time
            num_steps += 1
        
        print(f"Avg time per iteration: {total_time_taken*1000.0/num_steps} ms")

    def compute_test_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        # Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        #print(len(dataloader.dataset))
        scores = []
        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"])
            class_indices = torch.argmax(y_hat, dim=-1)
            for pred_index, gt_index in zip(class_indices, batch["labels"]):
                scores.append(pred_index == gt_index)
        print(f"Accuracy for {len(scores)} samples: {100*sum(scores)/len(scores)}%")
    
    def compute_test_avg_class_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        #Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        scores = []
        all_class_indices=[]
        all_batch_labels=[]
        for batch_idx, batch in tqdm(enumerate(dataloader),unit="batch"):
            y_hat = self.model(batch["frames"])
            class_indices = torch.argmax(y_hat, dim=-1)

            #print(len(class_indices))
            for i in range(len(batch["labels"])):
                all_batch_labels.append(batch["labels"][i])
                all_class_indices.append(class_indices[i])
            for pred_index, gt_index in zip(class_indices, batch["labels"]):
                scores.append(pred_index == gt_index)
        cm = confusion_matrix(np.array(all_batch_labels),np.array(all_class_indices))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Average Class Accuracy for {len(all_batch_labels)} samples: {np.mean(cm.diagonal())*100}%")
