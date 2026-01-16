import torch.nn as nn

# Choice of the loss function: https://arxiv.org/pdf/2312.05391
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_comp = 1 - pred
        target_comp = 1 - target
        return 1 - (pred * target).sum() / (pred + target).sum() - (pred_comp * target_comp).sum() / (pred_comp + target_comp).sum()