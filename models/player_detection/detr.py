import torch
from torch import optim
from torchtyping import TensorType
import pytorch_lightning as pl
from transformers import DetrImageProcessor, DetrForObjectDetection

from models.base.base_module import BaseModule
from losses.iou import IOULoss

class DeTR(BaseModule):
    def __init__(self, yolo_pth: str, loss_fn=IOULoss()):
        super().__init__(loss_fn)

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def forward(self, image: TensorType['batch', 1, 'row', 'column']):
        """
        Args:
            x (TensorType['batch', 1, 'row', 'column']): Batched images, single channel

        Returns:
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)