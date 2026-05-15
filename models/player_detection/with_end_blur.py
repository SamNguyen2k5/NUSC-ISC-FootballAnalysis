from models.base.unet_heatmap import UNetHeatmap
import torch
from torch import optim
from torch.nn import MSELoss, ModuleDict
import torch.nn.functional as TorchF
import torchvision.transforms.v2.functional as TorchVisionF
from torchtyping import TensorType

class UNetHeatmapWithEndBlur(UNetHeatmap):
    def forward(self, x: TensorType['batch', 1, 'row', 'column']) -> TensorType['batch', 1, 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 1, 'row', 'column']): Batched images, single channel
        """
        x = super().forward(x)
        x = TorchVisionF.gaussian_blur(x, kernel_size=[9, 9], sigma=[5.0, 5.0])
        
        # B, C, H, W = x.shape
        # x = x.view(B, C, H * W)
        # x = TorchF.softmax(x, dim=-1)
        # x = x.view(B, C, H, W)
        return x