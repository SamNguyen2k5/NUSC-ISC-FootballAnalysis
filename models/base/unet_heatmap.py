import torch
from torch import optim
from torchtyping import TensorType
import pytorch_lightning as pl

from losses.dice_loss import DiceLossWithGradient

class UNetHeatmap(pl.LightningModule):
    def __init__(self, eps=1e-6, loss_fn=DiceLossWithGradient()):
        super().__init__()
        self.unet = torch.hub.load('sm00thix/unet', 'unet', 
            pretrained=False, in_channels=1, out_channels=1,
            pad=True, bilinear=True, normalization='bn'
        )

        self.eps = eps
        self.loss_fn = loss_fn

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

        x = min_max_scale(x)
        x = self.unet(x)
        x = min_max_scale(x)
        return x
    
    def _step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.float()
        out_mask = self.forward(img)
        loss = self.loss_fn(out_mask, mask)
        return loss, out_mask, mask

    def training_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)