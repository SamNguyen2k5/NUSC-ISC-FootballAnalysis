# Numerical Coordinate Regression with Convolutional Neural Networks, Nibali et al, 2018

import torch
from tqdm import tqdm
from torchtyping import TensorType

from pipelines.player_detection.cropper import Cropper

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchtyping import TensorType
import torchvision.transforms.functional as F

import pytorch_lightning as pl

class CoordinateRegression(pl.LightningModule):
    def __init__(self, ):
        super().__init__()

    def forward(self, heatmap: TensorType['batch', 1, 'row', 'column']) -> TensorType['batch', 1, 'row', 'column']:
        raise NotImplementedError()

        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        B, C, H, W = heatmap.shape
        if C == 1:
            raise NotImplementedError("Only black and white heatmap.")

        y_base, x_base = torch.meshgrid(torch.arange(H), torch.arange(W))
        y_base = (2 * y_base - (H - 1)) / H
        x_base = (2 * x_base - (W - 1)) / W

        y_base = y_base.view(1, 1, H, W)
        x_base = x_base.view(1, 1, H, W)

        x_coords = x_base * heatmap
