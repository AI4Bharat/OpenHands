import torch
import torch.nn as nn
import torch.nn.functional as F


CrossEntropyLoss = nn.CrossEntropyLoss

class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smooth_factor=0.1):
        self.smooth_factor = smooth_factor

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smooth_factor) * nll_loss + self.smooth_factor * smooth_loss
        return loss