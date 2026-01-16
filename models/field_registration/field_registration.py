import pytorch_lightning as pl
from .segmentation import Segmentation

class FieldRegistration(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.segmentation = Segmentation()