from importlib import resources
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules.spiideo.spiideo import SpiideoDatasetArgs
from datamodules.spiideo.gaussian import Spiideo2DGaussianMaskDataModule

from datamodules.base_datamodule import BaseDataModuleArgs
from models.base.unet_heatmap import UNetHeatmap

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_2d_gaussian_data_module = Spiideo2DGaussianMaskDataModule(
    SpiideoDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule'])
)
unet_model = UNetHeatmap()

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(**config['trainer'], logger=wandb_logger)
trainer.fit(model=unet_model, datamodule=spiideo_2d_gaussian_data_module)