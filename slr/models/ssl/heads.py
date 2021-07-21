import torch.nn as nn
from torch.nn import functional as F

class BertMLMHead(nn.Module):
    def __init__(self, config, n_features):
        super(BertMLMHead, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, n_features)
                                      
    def forward(self, encoder_outputs):
        transformed = F.gelu(self.transform(encoder_outputs))
        transformed = self.layer_norm(transformed)
        
        output = self.output_layer(transformed)
        return output

class DirectionClassificationHead(nn.Module):
    def __init__(self, config, d_out_classes):
        super(DirectionClassificationHead, self).__init__()
        
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, d_out_classes)
    
    def forward(self, encoder_outputs):
        transformed = F.gelu(self.transform(encoder_outputs))
        transformed = self.layer_norm(transformed)
        
        output = self.output_layer(transformed)
        output = F.log_softmax(output, dim=-1)
        return output
