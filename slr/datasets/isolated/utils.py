import albumentations as A
from slr.datasets import video_transforms
from pytorchvideo import transforms as ptv_transforms


def _get_rgb_albumentation_transforms(img_resize_dims):
    return A.Compose([A.Resize(img_resize_dims[0], img_resize_dims[1])])


def _get_rgb_video_transforms():
    return [video_transforms.NumpyToTensor()]


def get_rgb_data_transforms(img_resize_dims):
    return video_transforms.Compose(
        [
            video_transforms.Albumentations2DTo3D(
                _get_rgb_albumentation_transforms(img_resize_dims)
            ),
        ]
        + _get_rgb_video_transforms()
    )


def get_data_transforms(modality, img_resize_dims):
    transforms = None
    if modality == "rgb" or modality == "rgbd":
        return get_rgb_data_transforms(img_resize_dims)

    # TODO: add default pose transforms
    return transforms
