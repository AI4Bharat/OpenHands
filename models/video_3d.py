import torch.nn as nn
from .utils import slow_r50

class ClassificationModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.backbone = slow_r50(pretrained=True)
        self.classifier = nn.Linear(400, num_class)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x