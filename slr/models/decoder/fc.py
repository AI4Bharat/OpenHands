import torch.nn as nn
import math

class FC(nn.Module):
    def __init__(
        self, n_features, num_class, dropout_ratio=0.2, **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x
