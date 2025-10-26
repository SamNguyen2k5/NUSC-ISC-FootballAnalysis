import cv2
import re, glob
import numpy as np
from torch.utils.data import Dataset
from itertools import pairwise
from .calibration import CalibrationDataset

class MaskingCalibrationDataset(CalibrationDataset):
    def __init__(self, img_folder, keys, width=None, height=None, centered=False, n_limit=None, mode='none', keypoint_only=False, transform=None, target_transform=None):
        super().__init__(img_folder, keys, width, height, centered, n_limit, mode, transform, target_transform)
        self.keypoint_only = keypoint_only

    @classmethod
    def from_folder(cls, img_folder, width=None, height=None, centered=False, n_limit=None, mode='none', keypoint_only=False, transform=None, target_transform=None):
        keys = [
            re.search(r'(\d+)\.jpg$', file).group(1) 
            for file in glob.glob(f'{img_folder}/*.jpg')
        ]
        return cls(img_folder, keys, width=width, height=height, centered=centered, n_limit=n_limit, mode=mode, keypoint_only=keypoint_only, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        mask_img = np.zeros(image.shape[:2])
        for name, coords in label.items():
            for coord in coords:
                cv2.circle(mask_img, coord, 2, (1., ), 2)
            if not self.keypoint_only:
                for coord_bef, coord_aft in pairwise(coords):
                    cv2.line(mask_img, coord_bef, coord_aft, (1., ), 2)

        return image, mask_img
