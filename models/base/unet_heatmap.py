import torch
from torch import optim
from torch.nn import MSELoss, ModuleDict
import torch.nn.functional as F
from torchmetrics import TotalVariation

from torchtyping import TensorType

from models.base.base_module import BaseModule
from losses.total_variation import TotalVariationLoss
from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss

class UNetHeatmap(BaseModule):
    def __init__(
        self, 
        loss_fns={'focal': FocalLoss(alpha=0.99, gamma=2)}, 
        lambdas=[1],
        unet_path=None, 
        eps=1e-8,
        lr=2e-4
    ):
        super().__init__(
            loss_fns=ModuleDict(loss_fns),
            lambdas=lambdas,
            log_small_losses=True
        )

        if unet_path is None:
            self.unet = torch.hub.load('sm00thix/unet', 'unet', 
                pretrained=False, in_channels=1, out_channels=1,
                pad=True, bilinear=True, normalization='bn'
            )

        self.eps = eps
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: TensorType['batch', 1, 'row', 'column']) -> TensorType['batch', 1, 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 1, 'row', 'column']): Batched images, single channel

        Returns:
            _type_: _description_
        """
        def min_max_scale(x):
            min_x = torch.amin(x, dim=(-1, -2), keepdim=True)
            max_x = torch.amax(x, dim=(-1, -2), keepdim=True)
            return (x - min_x + self.eps) / (max_x - min_x + self.eps)

        x = x.float()
        x = min_max_scale(x)
        x = self.unet(x)
        x = min_max_scale(x)
        return x
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)