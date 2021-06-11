import pytorch_lightning as pl
from omegaconf import OmegaConf
import torchvision
import albumentations as A
import hydra
import slr
from slr.datasets.transforms import Albumentations3D


class CommonDataModule(pl.LightningDataModule):
    # TODO: Valid, Test datasets
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.transforms = self.create_transform()
        self.dataset = None

    def prepare_data(self):
        self._instantiate_dataset()

    def setup(self, stage=None):
        self.dataset = self._instantiate_dataset()

    def train_dataloader(self):
        dataloader = hydra.utils.instantiate(
            self.data_cfg.dataloader, dataset=self.dataset
        )
        return dataloader

    def create_transform(self):
        albu_transforms = A.Compose(
            [
                *self.get_albumentations_transforms(),
            ]
        )

        transforms = torchvision.transforms.Compose(
            [Albumentations3D(albu_transforms), *self.get_video_transforms()]
        )
        return transforms

    def get_video_transforms(self):
        video_transforms_config = self.data_cfg.transforms.video
        video_transforms_config = OmegaConf.to_container(
            video_transforms_config, resolve=True
        )
        video_transforms = []
        for transform in video_transforms_config:
            for transform_name, transform_args in transform.items():
                if not transform_args:
                    transform_args = {}
                new_trans = getattr(slr.datasets.transforms, transform_name)(
                    **transform_args
                )
                video_transforms.append(new_trans)
        return video_transforms

    def get_albumentations_transforms(self):
        albu_config = self.data_cfg.transforms.albumentations
        if not albu_config:
            return []
        albu_config = OmegaConf.to_container(albu_config, resolve=True)
        albu_transforms = []
        for transform in albu_config:
            for transform_name, transform_args in transform.items():
                transform = A.from_dict(
                    {
                        "transform": {
                            "__class_fullname__": "albumentations.augmentations.transforms."
                            + transform_name,
                            **transform_args,
                        }
                    }
                )
                albu_transforms.append(transform)
        return albu_transforms

    def _instantiate_dataset(self):
        data_cfg = self.data_cfg
        transforms = self.transforms
        if getattr(data_cfg, "dataset", None):
            dataset = hydra.utils.instantiate(data_cfg.dataset, transforms=transforms)
        else:
            raise ValueError(f"{data_cfg.dataset} not found")
        return dataset
