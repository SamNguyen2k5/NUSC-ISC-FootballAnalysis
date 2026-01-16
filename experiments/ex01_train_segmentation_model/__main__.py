from importlib import resources
import yaml

import pytorch_lightning as pl
from datamodules.calibration.field_masking import (
    MaskingCalibrationDataModule, MaskingCalibrationDatasetArgs
)
from datamodules.base_datamodule import BaseDataModuleArgs
from models.field_registration import Segmentation

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)
    print(config)

field_masking_data_module = MaskingCalibrationDataModule(
    MaskingCalibrationDatasetArgs(**config['masking_calibration_dataset']), 
    BaseDataModuleArgs(**config['masking_calibration_datamodule'])
)
segmentation_model = Segmentation()

trainer = pl.Trainer(**config['trainer'])
trainer.fit(model=segmentation_model, datamodule=field_masking_data_module)