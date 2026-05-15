import torch
import torch.nn as nn
import torch.nn.functional as F
from transforms.sobel import sobel_magnitude

# https://akashprakas.github.io/akashBlog/posts/2022-08-24-understanding%20the%20losses%20in%20centernet%20architecture.html
# Code from: https://amaarora.github.io/posts/2020-06-29-FocalLoss.html

class FocalLoss(nn.Module):
    """_summary_
    A better loss function for heatmap prediction.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, alpha=.20, gamma=0.2):
        super().__init__()
        self.alpha = torch.tensor([1 - alpha, alpha])
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # 1. Flatten to 1D
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # 2. Clipping to ensure range for log (prevents NaNs)
        y_pred = torch.clamp(y_pred, min=1e-3, max=1 - 1e-3)

        # 3. Create the "Classed Tensors" correctly
        # Use torch.stack to keep the operation differentiable
        # Shape: [2, N] where row 0 is (1-p) and row 1 is (p)
        y_pred_stacked = torch.stack([1 - y_pred, y_pred])
        y_true_stacked = torch.stack([1 - y_true, y_true])
        
        # print('stacked y: ', y_pred_stacked.shape)
        # print('y_pred range: ', y_pred.min(), y_pred.max())
        # print('y_true range: ', y_true.min(), y_true.max())

        # 4. Move alpha to the correct device automatically
        alpha = self.alpha.to(y_pred.device).view(2, 1)
        # print('stacked alpha: ', alpha.shape)

        # 5. Focal Loss calculation
        # -(alpha) * (unselected_class_target)**gamma * log(selected_class_prediction)
        # Note: in binary focal loss, (1 - y_true) is the weight for the current class
        f_loss = - alpha * y_true_stacked * ((1 - y_pred_stacked) ** self.gamma) * torch.log(y_pred_stacked)

        # Return the mean of both classes
        return f_loss.mean()
