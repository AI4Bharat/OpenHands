import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorchvideo.transforms import transforms as ptv_transforms
import albumentations as A
import hydra
from ..datasets import pose_transforms, video_transforms

def create_pose_transforms(transforms_cfg):
    all_transforms = []
    for transform in transforms_cfg:
        for transform_name, transform_args in transform.items():
            if not transform_args:
                transform_args = {}
            new_trans = getattr(pose_transforms, transform_name)(**transform_args)
            all_transforms.append(new_trans)
    return pose_transforms.Compose(all_transforms)

class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._instantiate_dataset(self.data_cfg.train_pipeline)
            self.valid_dataset = self._instantiate_dataset(self.data_cfg.valid_pipeline)
            print("Train set size:", len(self.train_dataset))
            print("Valid set size:", len(self.valid_dataset))

            assert self.train_dataset.in_channels == self.valid_dataset.in_channels
            self.in_channels = self.valid_dataset.in_channels
            assert self.train_dataset.num_class == self.valid_dataset.num_class
            self.num_class = self.valid_dataset.num_class
        elif stage == "test":
            self.test_dataset = self._instantiate_dataset(self.data_cfg.test_pipeline)

            self.in_channels = self.test_dataset.in_channels
            self.num_class = self.test_dataset.num_class
        else:
            raise ValueError("Unknown `stage` value when calling `data_module.setup()`")

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
    
    def test_dataloader(self):
        dataloader = hydra.utils.instantiate(
            self.data_cfg.test_pipeline.dataloader,
            dataset=self.test_dataset,
            collate_fn=self.test_dataset.collate_fn,
        )
        return dataloader

    def create_video_transforms(self, transforms_cfg):
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
            if self.data_cfg.modality == "video":
                transforms = self.create_video_transforms(transforms_cfg)
            elif self.data_cfg.modality == "pose":
                transforms = create_pose_transforms(transforms_cfg)
            else:
                raise ValueError(f"{self.data_cfg.modality} modality not supported")
        else:
            transforms = None

        dataset_cfg = getattr(pipeline_cfg, "dataset", None)
        if dataset_cfg is None:
            raise ValueError(f"`dataset` section missing in pipeline")

        dataset = hydra.utils.instantiate(dataset_cfg, transforms=transforms)
        return dataset
