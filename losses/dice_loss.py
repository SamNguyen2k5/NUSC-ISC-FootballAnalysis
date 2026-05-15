import torch.nn as nn
from transforms.sobel import sobel_magnitude

# Choice of the loss function: https://arxiv.org/pdf/2312.05391
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_comp = 1 - pred
        target_comp = 1 - target
        return 1 - (pred * target).sum() / (pred + target).sum() - (pred_comp * target_comp).sum() / (pred_comp + target_comp).sum()

class DiceLossWithGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.lambda_d0 = 0.5
        self.lambda_d1 = 0.5

    def forward(self, pred, target):
        d_pred = sobel_magnitude(pred)
        d_target = sobel_magnitude(target)
        loss_0 = self.dice_loss.forward(pred, target) 
        loss_1 = self.dice_loss.forward(d_pred, d_target) 
        return self.lambda_d0 * loss_0 + self.lambda_d1 * loss_1