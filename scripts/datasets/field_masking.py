import cv2
import numpy as np
from torch.utils.data import Dataset
from itertools import pairwise
from .calibration import CalibrationDataset

class MaskingCalibrationDataset(CalibrationDataset):
    def __init__(self, img_folder, keys, mode='none', transform=None, target_transform=None):
        super().__init__(img_folder, keys, mode, transform, target_transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        mask_img = np.zeros(image.shape[:2])
        for name, coords in label.items():
            for coord in coords:
                cv2.circle(mask_img, coord, 3, (255, ), 3)
            for coord_bef, coord_aft in pairwise(coords):
                cv2.line(mask_img, coord_bef, coord_aft, (255, ), 4)

        return image, mask_img
