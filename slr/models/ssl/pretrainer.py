import torch.nn as nn
from .heads import BertMLMHead, DirectionClassificationHead

class PreTrainingModel(nn.Module):
    def __init__(self, base_encoder, n_features, config, use_direction=False, d_out_classes=0):
        super().__init__()
        self.bert = base_encoder
        self.mlm = BertMLMHead(config, n_features)
        self.use_direction = use_direction
        if self.use_direction:
            if not d_out_classes:
                raise ValueError("`d_out_classes` required")
            self.d_classifier = DirectionClassificationHead(config, d_out_classes)
        
    def forward(self, x):
        # (B, T, V, C)
        B,T,V,C = x.shape
        x = x.reshape(B,T, V*C)
        x = self.bert(x)
        mlm_outputs = self.mlm(x)
        
        d_outputs = None
        #direction
        if self.use_direction:
            d_outputs = self.d_classifier(x)
            d_outputs = d_outputs
            d_outputs = d_outputs.reshape(*d_outputs.shape[:2], 59, 4)
        return mlm_outputs, d_outputs
