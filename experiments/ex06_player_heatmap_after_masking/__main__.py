import torch, gc
gc.collect()
torch.mps.empty_cache()

from torch.nn import L1Loss

from importlib import resources
import wandb
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner

from datamodules.spiideo.gaussian import Spiideo2DGaussianMaskDataModule, Spiideo2DGaussianMaskDatasetArgs
from datamodules.base_datamodule import BaseDataModuleArgs

from models.base.unet_heatmap import UNetHeatmap
from models.player_detection.mask_then_heatmap import MaskThenHeatmap
from models.player_detection.unet_gaussian import UNetHeatmapGaussian

from callbacks.log_heatmap import LogHeatmapCallback
from callbacks.log_config import LogConfigArtifactCallback

from losses.focal_loss import FocalLoss
from losses.total_intensity import TotalIntensity
from losses.fp_aware_loss import FPAwareLoss

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_2d_gaussian_data_module = Spiideo2DGaussianMaskDataModule(
    Spiideo2DGaussianMaskDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule']),
)

masking_model = UNetHeatmap.load_from_checkpoint(config['pitch_cover_model'])
# unet_model = UNetHeatmap.load_from_checkpoint(config['gaussian_heatmap_model'])
unet_model = UNetHeatmap(in_channels=3, lr=0.005)
# segmentation_model = UNetHeatmap.load_from_checkpoint(config['pitch_segmentation_model'])
# unet_model = UNetHeatmap(
#     lr=1e-5,
#     loss_fns={
#         'focal': FocalLoss(alpha=0.99, gamma=2),
#         'intensity': TotalIntensity(),
#     }, 
#     lambdas=[0.99, 0.01],
#     in_channels=3
# )

# mask_then_heatmap_model = MaskThenHeatmap(
#     masking_model, unet_model, segmentation_model, 
#     alpha=0.99, gamma=2, lmbd_segment=0.01, lmbd_intensity=0.01
# )

unet_gaussian_model = UNetHeatmapGaussian(masking_model, unet_model)

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(
    **config['trainer'],
    # fast_dev_run=7,
    logger=wandb_logger, 
    callbacks=[
        LogConfigArtifactCallback(config), 
        LogHeatmapCallback(), 
    ]
)

trainer.fit(model=unet_gaussian_model, datamodule=spiideo_2d_gaussian_data_module)
# trainer.validate(model=unet_gaussian_model, datamodule=spiideo_2d_gaussian_data_module)