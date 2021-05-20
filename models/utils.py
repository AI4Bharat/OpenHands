import torch
import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet
from torch.hub import load_state_dict_from_url

checkpoint_paths = {
    "slow_r50": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth"
}

def reshape_conv_input_size(in_channels, model):
    "Change convolution layer to adopt to various input channels"
    assert in_channels == 1 or in_channels >= 4
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            break

    module.in_channels = in_channels
    weight = module.weight.detach()

    if in_channels == 1:
        module.weight = nn.parameter.Parameter(weight.sum(1, keepdim=True))
    else:
        curr_in_channels = module.weight.shape[1]
        to_concat = torch.Tensor(
            module.out_channels,
            module.in_channels - curr_in_channels,
            *module.kernel_size,
        )
        module.weight = nn.parameter.Parameter(
            torch.cat([module.weight, to_concat], axis=1)
        )

def slow_r50(in_channels=3, pretrained: bool = False, progress: bool = True, **kwargs):
    model = create_resnet(
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(8, 7, 7),
        model_depth=50,
        **kwargs,
    )

    if pretrained:
        path = checkpoint_paths["slow_r50"]
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)

    if in_channels != 3:
        reshape_conv_input_size(in_channels, model)

    return model