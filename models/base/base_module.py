import torch
from torch.nn import ModuleDict
from torch import optim
from torchtyping import TensorType
import pytorch_lightning as pl

class BaseModule(pl.LightningModule):
    def __init__(self, loss_fns, lambdas=None, log_small_losses=False):
        super().__init__()

        if not isinstance(loss_fns, ModuleDict):
            raise AttributeError("loss_fns should be of type torch.nn.ModuleDict({loss_fn_name: loss_fn}).")

        self.loss_fns = loss_fns

        if lambdas is None:
            lambdas = [1 / len(loss_fns) for _ in loss_fns]

        self.lambdas = lambdas
        self.log_smaller_losses = log_small_losses

    def forward(self, x):
        raise NotImplementedError()
    
    def _step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        losses = {loss_name: loss_fn(y_pred, y_true) for loss_name, loss_fn in self.loss_fns.items()}
        total_loss = sum(loss * coeff for loss, coeff in zip(losses.values(), self.lambdas))
        # [TODO]: possible gradient error here
        return total_loss, losses, y_pred, y_true

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