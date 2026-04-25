from typing import Tuple

import torch
import torch.nn.functional as TorchF
import torchvision.transforms.functional as TorchvisionF
from torchtyping import TensorType

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from models.base.unet_heatmap import UNetHeatmap
from models.player_detection.unet_gaussian import UNetHeatmapGaussian
from pipelines.player_detection.crop_then_stitch import CropThenStitch
from pipelines.player_detection.location_detector import LocationDetector

class PlayerDetector(pl.LightningModule):
    def __init__(
        self, 
        unet_gaussian_model: UNetHeatmapGaussian,
        pitch_cover: UNetHeatmap,
        location_detector: LocationDetector,
        keypoint_threshold: float = 0.995,
        debug: bool = False
    ):
        super().__init__()
        self.unet_gaussian_model = unet_gaussian_model.eval()
        self.pitch_cover = pitch_cover.eval()
        self.location_detector = location_detector.eval()
        self.keypoint_threshold = keypoint_threshold
        self.debug = debug

    def forward(self, img: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['n_components', 2]:
        B, C, H, W = img.shape
        if B != 1:
            raise AttributeError("Only single-batched images are supported.")
        if C != 3:
            raise AttributeError("Only multi-colour images are supported. If the images are single-channel, convert to rgb before use.")

        with torch.no_grad():
            cover = self.pitch_cover(img)
            player_mask_all = self.unet_gaussian_model(img) * cover

            if self.debug:
                plt.imshow(player_mask_all[0].permute((1, 2, 0)).detach().cpu().float())
                plt.show()
                print('player_mask_all.shape = ', player_mask_all.shape)

            clipped_mask_all = player_mask_all.mean(axis=1).unsqueeze(1)
            clipped_mask_all[clipped_mask_all < self.keypoint_threshold] = 0.

            if self.debug:
                plt.imshow(clipped_mask_all[0].permute((1, 2, 0)).detach().cpu().float())
                plt.show()
                print('clipped_mask_all.shape = ', clipped_mask_all.shape)

            keypoints = self.location_detector(clipped_mask_all)
            return keypoints