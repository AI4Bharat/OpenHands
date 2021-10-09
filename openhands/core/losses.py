import torch
import torch.nn as nn
import torch.nn.functional as F


CrossEntropyLoss = nn.CrossEntropyLoss

class SmoothedCrossEntropyLoss(nn.Module):
    """
    Calculates Cross-entropy loss with label smoothing.

    Args:
        smooth_factor (float): label smoothing regularization coefficient
    """
    def __init__(self, smooth_factor=0.01):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smooth_factor = smooth_factor

    def forward(self, input, target):
        """
        Calculate label smoothed cross entropy loss

        Args:
            input (torch.Tensor):  :math:`(N, C)` 
            target (torch.Tensor): :math:`(N)` 

            where 
            :math:`N` = Batch Size,
            :math:`C` = number of classes.
            
            
        Returns:
            torch.Tensor: Calulated cross entropy loss with label smoothing.
        """
        log_probs = F.log_softmax(input, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smooth_factor) * nll_loss + self.smooth_factor * smooth_loss
        return loss.mean()
