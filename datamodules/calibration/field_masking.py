from typing import List, Literal
from itertools import pairwise
import warnings
from pydantic import BaseModel

import cv2
import torch
import numpy as np

from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs
from .calibration import CalibrationDataset, CalibrationDatasetArgs

class MaskingCalibrationDatasetArgs(CalibrationDatasetArgs, BaseModel):
    fill_mode: Literal['keypoint', 'line', 'fill'] = 'line'
    excludes: List[int] = []
    in_channels: int = 1

class MaskingCalibrationDataset(CalibrationDataset):
    def __init__(self, args: MaskingCalibrationDatasetArgs, mode: str):
        super().__init__(args, mode)
        self.args = args

    @classmethod
    def from_folder(cls, **kwargs):
        warnings.warn('Depercated! Use the default constructor instead.')

    def __getitem__(self, idx):
        image, labels = super().__getitem__(idx)

        if self.args.in_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.args.in_channels != 3:
            raise NotImplementedError("Function not supported for images with {self.args.in_channels} colour channels")

        mask_img = np.zeros(image.shape, dtype='float32')
        for label, coords in labels.items():
            if label in self.args.excludes:
                continue

            for coord in coords:
                cv2.circle(mask_img, coord, 2, (1., ), 2)

            if self.args.fill_mode != 'keypoint':
                print(coords)
                for coord_bef, coord_aft in pairwise(coords):
                    cv2.line(mask_img, coord_bef, coord_aft, (1., ), 2)

        if self.args.in_channels == 1:
            image = torch.tensor(image).unsqueeze(0)
        elif self.args.in_channels == 3:
            image = torch.tensor(image).permute((2, 0, 1))

        mask_img = torch.tensor(mask_img).unsqueeze(0)
        return image, mask_img

class MaskingCalibrationDataModule(BaseDataModule):
    def __init__(self, args_dataset: MaskingCalibrationDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either train, valid or test.')

        return MaskingCalibrationDataset(self.args_dataset, mode=mode)

if __name__ == '__main__':
    ds_args = MaskingCalibrationDatasetArgs(
        img_folder='data/calibration',
        width=320, height=180
    )
    dm_args = BaseDataModuleArgs(batch_size=16)
    dm = MaskingCalibrationDataModule(ds_args, dm_args)
    dm.setup('fit')
    print(next(iter(dm.train_dataloader())))
