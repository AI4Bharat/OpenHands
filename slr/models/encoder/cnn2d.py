import torch
import torch.nn as nn
import timm

class CNN2D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        backbone="resnet18",
        pretrained=True
    ):
        super().__init__()
        assert in_channels == 3
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.n_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.shape
        cnn_embeds = []
        for i in range(t):
            out = self.backbone(x[:, :, i, :, :])
            out = out.view(out.shape[0], -1)
            cnn_embeds.append(out)

        return torch.stack(cnn_embeds, dim=0)
