import torch
import torch.nn as nn


def reshape_conv_input_size(in_channels, model):
    """
    Change convolution layer to adopt to various input channels
    """
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
