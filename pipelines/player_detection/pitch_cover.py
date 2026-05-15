from models.base.unet_heatmap import UNetHeatmap

import pytorch_lightning as pl

import torch
from torch import optim
from torch.nn import MSELoss, ModuleDict
import torch.nn.functional as TorchF
import torchvision.transforms.v2.functional as TorchvisionF
from torchtyping import TensorType

class PitchCover(pl.LightningModule):
    def __init__(self, unet, pad_size=31, **kwargs):
        super().__init__(**kwargs)
        self.unet = unet.eval()
        self.pad_size = pad_size

    def forward(self, x: TensorType['batch', 1, 'row', 'column']) -> TensorType['batch', 1, 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 1, 'row', 'column']): Batched images, single channel
        """
        x = self.unet(x)
        x = TorchvisionF.gaussian_blur(x, kernel_size=self.pad_size)
        x = TorchF.pad(x, pad=(self.pad_size // 2,) * 4, mode='replicate')
        x = TorchF.max_pool2d(x, kernel_size=self.pad_size, stride=1)
        return x