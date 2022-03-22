import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from ..core.data import DataModule
from ..models.loader import get_model

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
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if stage == "test":
            self.model.to(self._device).eval()
    
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
            y_hat = self.model(batch["frames"].to(self._device)).cpu()

            class_indices = torch.argmax(y_hat, dim=-1)

            for i, pred_index in enumerate(class_indices):
                # label = self.datamodule.test_dataset.id_to_gloss[pred_index]
                label = dataloader.dataset.id_to_gloss[pred_index]
                filename = batch["files"][i]
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
        dataset_scores, class_scores = {}, {}
        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)
            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):

                dataset_name = batch["dataset_names"][i]
                score = pred_index == gt_index
                
                if dataset_name not in dataset_scores:
                    dataset_scores[dataset_name] = []
                dataset_scores[dataset_name].append(score)

                if gt_index not in class_scores:
                    class_scores[gt_index] = []
                class_scores[gt_index].append(score)
        
        
        for dataset_name, score_array in dataset_scores.items():
            dataset_accuracy = sum(score_array)/len(score_array)
            print(f"Accuracy for {len(score_array)} samples in {dataset_name}: {dataset_accuracy*100}%")


        classwise_accuracies = {class_index: sum(scores)/len(scores) for class_index, scores in class_scores.items()}
        avg_classwise_accuracies = sum(classwise_accuracies.values()) / len(classwise_accuracies)

        print(f"Average of class-wise accuracies: {avg_classwise_accuracies*100}%")
