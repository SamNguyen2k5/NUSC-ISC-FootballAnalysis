import cv2
import numpy as np
from torch.utils.data import Dataset
from itertools import pairwise
from .calibration import CalibrationDataset

class MaskingCalibrationDataset(CalibrationDataset):
    n_limit: int

    def __init__(self, img_folder, keys, width=None, height=None, n_limit=None, mode='none', transform=None, target_transform=None):
        super().__init__(img_folder, keys, width, height, mode, transform, target_transform)
        self.n_limit = n_limit

    @classmethod
    def from_folder(cls, img_folder, width=None, height=None, n_limit=None, mode='none', transform=None, target_transform=None):
        ds = super().from_folder(img_folder, width, height, mode, transform, target_transform)
        ds.n_limit = n_limit
        return ds

    def __len__(self):
        if self.n_limit:
            return self.n_limit
        return super().__len__()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        mask_img = np.zeros(image.shape[:2])
        for name, coords in label.items():
            for coord in coords:
                cv2.circle(mask_img, coord, 2, (1., ), 2)
            for coord_bef, coord_aft in pairwise(coords):
                cv2.line(mask_img, coord_bef, coord_aft, (1., ), 2)

        return image, mask_img
