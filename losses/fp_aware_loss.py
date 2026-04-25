import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.focal_loss import FocalLoss

class FPAwareLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, alpha=.20, gamma=0.2, lmbd=0.):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.lmbd = lmbd

    def forward(self, y_pred, y_true, y_critical):
        return self.focal_loss(y_pred, y_true) + self.lmbd * self.focal_loss(y_pred, 1 - y_critical)