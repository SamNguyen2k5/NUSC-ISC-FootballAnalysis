import base64
from pathlib import Path
import yaml
import numpy as np
import torch
import litserve as ls

from datamodules.base_datamodule import BaseDataModuleArgs
from datamodules.spiideo import SpiideoDataModule, SpiideoDatasetArgs
from apis.payload import NumpyPayload
from apis.inference.base.image import ToImageAPIMixin

with Path(__file__).parent.joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)
    print(config)

class SpiideoVisualiserAPI(ToImageAPIMixin, ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.spiideo_dataset = SpiideoDataModule(
            SpiideoDatasetArgs(**config['datasets']['spiideo']), 
            BaseDataModuleArgs(**config['datasets']['spiideo_datamodule'])
        ).dataset('mini')

    def predict(self, x: any) -> torch.Tensor:
        if 0 <= x < len(self.spiideo_dataset):
            return self.spiideo_dataset[x]

        return None