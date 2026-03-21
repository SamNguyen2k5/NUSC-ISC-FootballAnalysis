from importlib import resources
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules.spiideo.gaussian import Spiideo2DGaussianMaskDataModule, Spiideo2DGaussianMaskDatasetArgs
from datamodules.base_datamodule import BaseDataModuleArgs
from models.base.unet_heatmap import UNetHeatmap
from losses.focal_loss import FocalLoss

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_2d_gaussian_data_module = Spiideo2DGaussianMaskDataModule(
    Spiideo2DGaussianMaskDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule']),
    debug=True
)
unet_model = UNetHeatmap(loss_fn=FocalLoss(alpha=0.9, gamma=1e-4))

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(**config['trainer'], logger=wandb_logger)
trainer.fit(model=unet_model, datamodule=spiideo_2d_gaussian_data_module)