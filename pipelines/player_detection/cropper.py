from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchtyping import TensorType
import torchvision.transforms.functional as F

import pytorch_lightning as pl

class Cropper(pl.LightningModule):
    def __init__(self, model: pl.LightningModule, window_crop: Tuple[int, int], batch_size_per_image: int = 4):
        super().__init__()
        self.model = model.eval()
        self.window_crop = window_crop
        self.batch_size_per_image = batch_size_per_image

    def _get_tiled_batch(self, image: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        H_crop, W_crop = self.window_crop   # pylint: disable=invalid-name
        B, C, H, W = image.shape            # pylint: disable=invalid-name

        if B != 1:
            raise NotImplementedError("Only supported for single-batched image.")
        if C != 1:
            raise NotImplementedError("Only black and white images.")
        
        # Calculate number of steps to cover the image with 50% stride (half-tile overlap)
        num_h = 1 + int(np.ceil((H - H_crop) / (H_crop / 2)))
        num_w = 1 + int(np.ceil((W - W_crop) / (W_crop / 2)))

        # Coordinates of the CENTERS of the trusted regions
        y_coords = np.linspace(H_crop / 2, H - H_crop / 2, num_h, dtype=int)
        x_coords = np.linspace(W_crop / 2, W - W_crop / 2, num_w, dtype=int)

        print(W_crop / 2, W - W_crop / 2, x_coords)

        tiles = []
        offsets = [] 

        for y in y_coords:
            for x in x_coords:
                tile = F.crop(image, y - H_crop // 2, x - W_crop // 2, H_crop, W_crop)
                tiles.append(tile)
                offsets.append((y - H_crop // 4, x - W_crop // 4, 
                                y + H_crop // 4, x + W_crop // 4))
                
        return torch.cat(tiles, dim=0), torch.tensor(offsets)

    def _for_each_piece_with_result(self, image: TensorType['batch', 'channel', 'row', 'column']):
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        img_batch, offset_batch = self._get_tiled_batch(image)
        H_crop, W_crop = self.window_crop   # pylint: disable=invalid-name

        img_batchloader = DataLoader(img_batch, batch_size=self.batch_size_per_image)
        offset_batchloader = DataLoader(offset_batch, batch_size=self.batch_size_per_image)

        for img_pieces, offset_pieces in zip(img_batchloader, offset_batchloader):
            with torch.no_grad():
                output_pieces = self.model(img_pieces)
                y0, y1 = H_crop // 4, 3 * H_crop // 4
                x0, x1 = W_crop // 4, 3 * W_crop // 4
                output_pieces = output_pieces[:, :, y0:y1, x0:x1]
                yield output_pieces, offset_pieces

    def forward(self, image: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        raise NotImplementedError()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)