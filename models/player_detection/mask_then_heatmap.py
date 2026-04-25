from models.base.composite_module import CompositeModule
import torch
from torch import optim
from torch.nn import MSELoss, ModuleDict
import torch.nn.functional as TorchF
import torchvision.transforms.v2.functional as TorchVisionF
from torchtyping import TensorType
from models.base.unet_heatmap import UNetHeatmap

from losses.focal_loss import FocalLoss
from losses.focal_loss_weighted import WeightedFocalLoss
from losses.fp_aware_loss import FPAwareLoss
from losses.total_intensity import TotalIntensity

class MaskThenHeatmap(CompositeModule):
    def __init__(self, masking_unet: UNetHeatmap, heatmap_unet: UNetHeatmap, segment_unet: UNetHeatmap, alpha=0.99, gamma=2, lmbd_segment=0.1, lmbd_intensity=0.05):
        super().__init__(log_small_losses=True)
        self.masking_unet = masking_unet.eval()
        self.heatmap_unet = heatmap_unet.train()
        self.segment_unet = segment_unet.eval()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.focal_loss_weighted = WeightedFocalLoss(alpha, gamma)
        self.intensity_loss = TotalIntensity()
        self.lmbd_segment = lmbd_segment
        self.lmbd_intensity = lmbd_intensity
        self.save_hyperparameters()

    def on_train_start(self):
        for net in [self.masking_unet, self.segment_unet]:
            for param in net.parameters():
                param.requires_grad = False

    def forward(self, x: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Batched images, single/multi-channel
        """
        x = x * self.masking_unet(x)
        y = self.heatmap_unet(x)
        y_segment = self.segment_unet(TorchVisionF.rgb_to_grayscale(x))
        return y, y_segment

    def predict_step(self, batch, batch_idx):
        return self(batch)[0]
    
    def _step(self, batch: TensorType['batch', 'channel', 'row', 'column'], batch_idx):
        try:
            x, y_true = batch
            y_pred, y_segment = self.forward(x)

            losses = {
                'vs_true': self.focal_loss(y_pred, y_true),
                'vs_segment': self.focal_loss_weighted(y_pred, 1 - y_segment, y_segment),
                'intensity': self.intensity_loss(y_pred, y_true)
            }
            lambdas = [1, self.lmbd_segment, self.lmbd_intensity]
            total_loss = sum(loss * coeff for loss, coeff in zip(losses.values(), lambdas))

            # [TODO]: possible gradient error here
            return total_loss, losses, y_pred, y_true

        except Exception as e:
            print('---- [Diagnosis] ----')
            print('x.shape = ', x)
            print('y_pred.shape = ', y_pred.shape)
            print('y_true.shape = ', y_true.shape)
            print('y_segment.shape = ', y_segment.shape)
            raise e
            
    
    def configure_optimizers(self):
        return optim.Adam(self.heatmap_unet.parameters(), lr=self.heatmap_unet.lr)