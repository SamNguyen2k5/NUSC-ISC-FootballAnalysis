import torch.nn as nn
from torchmetrics.image import TotalVariation

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tv_loss = TotalVariation(reduction='sum')

    def forward(self, y_pred, y_true):
        return self.tv_loss(y_pred) / y_pred.numel()