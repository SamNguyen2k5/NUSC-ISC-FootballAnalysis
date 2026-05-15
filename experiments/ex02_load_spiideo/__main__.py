from importlib import resources
import yaml

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules.spiideo import SpiideoDataModule, SpiideoDatasetArgs
from datamodules.base_datamodule import BaseDataModuleArgs

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_data_module = SpiideoDataModule(
    SpiideoDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule'])
)

ds = spiideo_data_module.dataset('mini')
print(ds[0])