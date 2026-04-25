from torch.nn import L1Loss, MSELoss

from importlib import resources
import wandb
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner

from datamodules.base_datamodule import BaseDataModuleArgs
from models.base.unet_heatmap import UNetHeatmap
from models.player_detection.with_end_blur import UNetHeatmapWithEndBlur

from callbacks.log_heatmap import LogHeatmapCallback
from callbacks.log_config import LogConfigArtifactCallback

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from losses.total_intensity import TotalIntensity

from datamodules.calibration.field_cover import CoverCalibrationDataModule, CoverCalibrationDatasetArgs, CoverCalibrationDataModule
from datamodules.base_datamodule import BaseDataModuleArgs

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

pitch_cover_calibration_datamodule = CoverCalibrationDataModule(
    CoverCalibrationDatasetArgs(**config['pitch_calibration_dataset']),
    BaseDataModuleArgs(**config['pitch_calibration_datamodule'])
)

unet_model = UNetHeatmap(
    lr=1e-4,
    loss_fns={
        # 'mse': MSELoss(),
        # 'intensity': TotalIntensity(),
        'dice': DiceLoss(),
    }, 
    lambdas=[1],
    in_channels=3,
)

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(
    **config['trainer'],
    logger=wandb_logger, 
    callbacks=[
        LogConfigArtifactCallback(config), 
        LogHeatmapCallback(), 
    ]
)

trainer.fit(model=unet_model, datamodule=pitch_cover_calibration_datamodule)