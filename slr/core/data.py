import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorchvideo.transforms import transforms as ptv_transforms
import albumentations as A
import hydra


class CommonDataModule(pl.LightningDataModule):
    # TODO: Test datasets
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

    def setup(self, stage=None):
        self.train_dataset = self._instantiate_dataset(self.data_cfg.train_pipeline)
        self.valid_dataset = self._instantiate_dataset(self.data_cfg.valid_pipeline)

    def train_dataloader(self):
        dataloader = hydra.utils.instantiate(
            self.data_cfg.train_pipeline.dataloader,
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = hydra.utils.instantiate(
            self.data_cfg.valid_pipeline.dataloader,
            dataset=self.valid_dataset,
            collate_fn=self.valid_dataset.collate_fn,
        )
        return dataloader

    def create_transform(self, transforms_cfg):
        albumentation_transforms = A.Compose(
            [
                *self.get_albumentations_transforms(transforms_cfg),
            ]
        )

        transforms = video_transforms.Compose(
            [
                video_transforms.Albumentations2DTo3D(albumentation_transforms),
                *self.get_video_transforms(transforms_cfg),
                *self.get_pytorchvideo_transforms(transforms_cfg),
            ]
        )
        return transforms

    def get_video_transforms(self, transforms_cfg):
        transforms = []
        video_transforms_config = transforms_cfg.video
        if not video_transforms_config:
            return transforms
        video_transforms_config = OmegaConf.to_container(
            video_transforms_config, resolve=True
        )
        for transform in video_transforms_config:
            for transform_name, transform_args in transform.items():
                if not transform_args:
                    transform_args = {}
                new_trans = getattr(video_transforms, transform_name)(**transform_args)
                transforms.append(new_trans)
        return transforms

    def get_pytorchvideo_transforms(self, transforms_cfg):
        transforms = []
        video_transforms_config = transforms_cfg.pytorchvideo
        if not video_transforms_config:
            return transforms
        video_transforms_config = OmegaConf.to_container(
            video_transforms_config, resolve=True
        )
        for transform in video_transforms_config:
            for transform_name, transform_args in transform.items():
                if not transform_args:
                    transform_args = {}
                new_trans = getattr(ptv_transforms, transform_name)(**transform_args)
                transforms.append(new_trans)
        return transforms

    def get_albumentations_transforms(self, transforms_cfg):
        transforms = []
        albu_config = transforms_cfg.albumentations
        if not albu_config:
            return transforms
        albu_config = OmegaConf.to_container(albu_config, resolve=True)
        
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
                transforms.append(transform)
        return transforms

    def _instantiate_dataset(self, pipeline_cfg):
        transforms_cfg = pipeline_cfg.transforms
        if transforms_cfg:
            transforms = self.create_transform(transforms_cfg)
        else:
            transforms = None

        dataset_cfg = getattr(pipeline_cfg, "dataset", None)
        if dataset_cfg is None:
            raise ValueError(f"{dataset_cfg} not found")

        dataset = hydra.utils.instantiate(dataset_cfg, transforms=transforms)
        return dataset
