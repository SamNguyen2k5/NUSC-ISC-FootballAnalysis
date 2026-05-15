from itertools import pairwise
from typing import List
import warnings
from pydantic import BaseModel

import cv2
import torch
import numpy as np

from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs
from .field_pitch import PitchCalibrationDatasetArgs, PitchCalibrationDataset

class CoverCalibrationDatasetArgs(PitchCalibrationDatasetArgs):
    in_channels: int = 1

class CoverCalibrationDataset(PitchCalibrationDataset):
    def __init__(self, args: CoverCalibrationDatasetArgs, mode: str):
        super().__init__(args, mode)
        self.args = args

    @classmethod
    def from_folder(cls, **kwargs):
        warnings.warn('Depercated! Use the default constructor instead.')

    def __getitem__(self, idx):
        image, pitch = super().__getitem__(idx)

        if self.args.in_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = torch.tensor(image).unsqueeze(0)
        elif self.args.in_channels == 3:
            image = torch.tensor(image).permute((2, 0, 1))
        else:
            raise NotImplementedError("Function not supported for images with {self.args.in_channels} colour channels")

        field_mask = pitch.plot_filled()
        field_mask = torch.tensor(field_mask).unsqueeze(0)
        return image, field_mask

class CoverCalibrationDataModule(BaseDataModule):
    def __init__(self, args_dataset: CoverCalibrationDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either train, valid or test.')

        return CoverCalibrationDataset(self.args_dataset, mode=mode)
