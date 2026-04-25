import torch
from torch import optim
from torch.nn import MSELoss, ModuleDict
import torch.nn.functional as TorchF
import torchvision.transforms.v2.functional as TorchVisionF
from torchtyping import TensorType

from models.base.composite_module import CompositeModule
from models.base.unet_heatmap import UNetHeatmap
from pipelines.player_detection.gaussianiser import HeatmapGaussianiser

from losses.focal_loss import FocalLoss
from losses.total_intensity import TotalIntensity

class UNetHeatmapGaussian(CompositeModule):
    def __init__(
        self, masking_unet: UNetHeatmap, heatmap_unet: UNetHeatmap, 
        alpha=0.99, gamma=2, 
        lmbd_intensity=0.05, lmbd_gaussianiser=0.1,
        top_k=1000, select_k=100, max_sigma=4
    ):
        super().__init__(log_small_losses=True)
        self.masking_unet = masking_unet.eval()
        self.heatmap_unet = heatmap_unet.train()
        self.gaussianiser = HeatmapGaussianiser(top_k=top_k, select_k=select_k, max_sigma=max_sigma)

        self.focal_loss = FocalLoss(alpha, gamma)
        self.intensity_loss = TotalIntensity()
        self.lmbd_intensity = lmbd_intensity
        self.lmbd_gaussianiser = lmbd_gaussianiser

        self.save_hyperparameters()

    def on_train_start(self):
        for net in [self.masking_unet]:
            for param in net.parameters():
                param.requires_grad = False

    def forward(self, x: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Batched images, single/multi-channel
        """
        # mask = self.masking_unet(x)
        # x = self.heatmap_unet(x * mask)
        x = self.heatmap_unet(x)
        x = (1 - self.lmbd_gaussianiser) * x + self.lmbd_gaussianiser * self.gaussianiser(x)
        return x

    def _step(self, batch: TensorType['batch', 'channel', 'row', 'column'], batch_idx):
        try:
            x, y_true = batch
            y_pred = self.forward(x)

            losses = {
                'vs_true': self.focal_loss(y_pred, y_true),
                'intensity': self.intensity_loss(y_pred, y_true)
            }
            lambdas = [1, self.lmbd_intensity]
            total_loss = sum(loss * coeff for loss, coeff in zip(losses.values(), lambdas))

            # [TODO]: possible gradient error here
            return total_loss, losses, y_pred, y_true

        except Exception as e:
            print('---- [Diagnosis] ----')
            print('x.shape = ', x)
            print('y_pred.shape = ', y_pred.shape)
            print('y_true.shape = ', y_true.shape)
            raise e
            
    
    def configure_optimizers(self):
        return optim.Adam(self.heatmap_unet.parameters(), lr=self.heatmap_unet.lr)