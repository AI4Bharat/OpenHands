import pytorchvideo
from .utils import reshape_conv_input_size

available_3d_backbones = [
    "slow_r50",
    "slowfast_r50",
    "slowfast_r101",
    "x3d_xs",
    "x3d_s",
    "x3d_m",
]


def get_3d_backbone(
    name, in_channels=3, pretrained: bool = False, progress: bool = True, **kwargs
):
    assert name in available_3d_backbones, "Please use any bonebone from " + str(
        available_3d_backbones
    )
    model = getattr(pytorchvideo.models.hub, name)(
        pretrained=pretrained, progress=progress, **kwargs
    )
    if in_channels != 3:
        reshape_conv_input_size(in_channels, model)

    return model