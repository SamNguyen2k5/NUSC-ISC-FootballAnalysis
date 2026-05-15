from importlib import resources
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules.calibration.field_masking import (
    MaskingCalibrationDataModule, MaskingCalibrationDatasetArgs
)
from datamodules.base_datamodule import BaseDataModuleArgs
from models.field_registration import Segmentation

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

field_masking_data_module = MaskingCalibrationDataModule(
    MaskingCalibrationDatasetArgs(**config['masking_calibration_dataset']), 
    BaseDataModuleArgs(**config['masking_calibration_datamodule'])
)
segmentation_model = Segmentation()

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(**config['trainer'], logger=wandb_logger)
trainer.fit(model=segmentation_model, datamodule=field_masking_data_module)