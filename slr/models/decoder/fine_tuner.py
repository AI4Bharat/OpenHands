import torch
import torch.nn as nn
import math


class FineTuner(nn.Module):
    def __init__(self, n_features, num_class, dropout_ratio=0.2, pooling_type=None):
        super().__init__()
        self.pooling_type = pooling_type
        if self.pooling_type == "att":
            from .utils import AttentionBlock

            self.attn_block = AttentionBlock(n_features)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):

        if self.pooling_type == "cls":
            x = x[:, 0]
        elif self.pooling_type == "max":
            x = torch.max(x, dim=1).values
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_type == "att":
            x = self.attn_block(x)

        x = self.dropout(x)
        x = self.classifier(x)
        return x
