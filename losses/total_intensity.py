import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalIntensity(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super().__init__()
        self.loss = loss

    def forward(self, y_pred, y_true):
        return self.loss(y_pred.mean(), y_true.mean())