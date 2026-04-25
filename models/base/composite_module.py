import torch
from torch.nn import ModuleDict
from torch import optim
from torchtyping import TensorType
import pytorch_lightning as pl

class CompositeModule(pl.LightningModule):
    def __init__(self, log_small_losses=False):
        super().__init__()
        self.log_smaller_losses = log_small_losses
        self.save_hyperparameters()

    def on_train_start(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()
    
    def _step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        # return total_loss, losses, y_pred, y_true
        raise NotImplementedError()

    def _log_losses(self, mode, total_loss, smaller_losses):
        self.log(f'{mode}_loss', total_loss, on_epoch=True, prog_bar=True)

        if self.log_smaller_losses:
            for loss_name, loss_value in smaller_losses.items():
                self.log(f'{mode}_{loss_name}', loss_value, on_epoch=True, prog_bar=True)

        return total_loss

    def training_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        total_loss, smaller_losses, _, _ = self._step(batch, batch_idx)
        self._log_losses("training", total_loss, smaller_losses)
        return total_loss

    def validation_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        total_loss, smaller_losses, _, _ = self._step(batch, batch_idx)
        self._log_losses("validation", total_loss, smaller_losses)
        return total_loss

    def test_step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        total_loss, smaller_losses, _, _ = self._step(batch, batch_idx)
        self._log_losses("test", total_loss, smaller_losses)
        return total_loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)