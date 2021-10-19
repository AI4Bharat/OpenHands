import torch
import torch.nn as nn
import pytorchvideo

AVAILABLE_3D_BACKBONES = [
    "i3d_r50",
    "c2d_r50",
    "csn_r101",
    "r2plus1d_r50",
    "slow_r50",
    "slowfast_r50",
    "slowfast_r101",
    "slowfast_16x8_r101_50_50",
    "x3d_xs",
    "x3d_s",
    "x3d_m",
    "x3d_l",
]


class CNN3D(nn.Module):
    """
    Initializes the 3D Convolution backbone. 
    
    **Supported Backbones**
    
    - `i3d_r50`
    - `c2d_r50`
    - `csn_r101`
    - `r2plus1d_r5`
    - `slow_r50`
    - `slowfast_r50`
    - `slowfast_r101`
    - `slowfast_16x8_r101_50_50`
    - `x3d_xs`
    - `x3d_s`
    - `x3d_m`
    - `x3d_l`
    
    Args:
        in_channels (int): Number of input channels 
        backbone (string): Backbone to use
        pretrained (bool, optional): Whether to use pretrained Backbone.  Default: ``True``
        **kwargs (optional): Will be passed to pytorchvideo.models.hub models;
    
    """
    def __init__(self, in_channels, backbone, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = self.get_3d_backbone(
            backbone, in_channels, pretrained, **kwargs
        )
        self.n_out_features = 400  # list(self.backbone.modules())[-2].out_features

    def forward(self, x):
        """
        forward step
        """
        x = self.backbone(x)
        return x.transpose(0, 1) # Batch-first

    def get_3d_backbone(
        self,
        name,
        in_channels=3,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs
    ):
        assert name in AVAILABLE_3D_BACKBONES, "Please use any bonebone from " + str(
            AVAILABLE_3D_BACKBONES
        )
        import pytorchvideo.models.hub as ptv_hub

        model = getattr(ptv_hub, name)(
            pretrained=pretrained, progress=progress, **kwargs
        )
        if in_channels != 3:
            reshape_conv_input_size(in_channels, model)

        return model


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
