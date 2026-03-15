import pytorch_lightning as pl
from .segmentation import Segmentation

class FieldRegistration(pl.LightningModule):
    def __init__(self):
        raise NotImplementedError()
        super().__init__()
        self.segmentation = Segmentation()