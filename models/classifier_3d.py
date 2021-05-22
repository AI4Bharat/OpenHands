import torch.nn as nn
from backones.pytorchvideo_3d import get_3d_backbone


class ClassificationModel3d(nn.Module):
    def __init__(
        self, in_channels, num_class, backbone_model, pretrained=True, dropout_ratio=0.2, **kwargs
    ):
        super().__init__()
        self.backbone = get_3d_backbone(backbone_model, in_channels, pretrained, **kwargs)
        n_channels_out = 400 # list(self.backbone.modules())[-2].out_features
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.classifier = nn.Linear(n_channels_out, num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x