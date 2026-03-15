from torch import optim
from torchtyping import TensorType
import pytorch_lightning as pl

class BaseModule(pl.LightningModule):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x):
        raise NotImplementedError()
    
    def _step(self, batch: TensorType['batch', 1, 'row', 'column'], batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

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