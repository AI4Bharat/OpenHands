import torch.nn as nn

class FC(nn.Module):
    def __init__(
        self, n_features, num_class, dropout_ratio=0.2, **kwargs
    ):
        super().__init__()
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.classifier = nn.Linear(n_features, num_class)

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x
