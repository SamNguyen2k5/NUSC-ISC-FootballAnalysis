import re, glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from itertools import pairwise
from .calibration import CalibrationDataset
from ..classes.pitch import Pitch

class PitchCalibrationDataset(CalibrationDataset):
    # def __init__(self, img_folder, keys, width=None, height=None, n_limit=None, mode='none', transform=None, target_transform=None):
    #     super().__init__(img_folder, keys, width, height, n_limit, mode, transform, target_transform)

    def __getitem__(self, idx):
        image, labels = super().__getitem__(idx)
        pitch = Pitch.from_one_annotation(labels)
        return image, pitch.keypoints

class PitchKeypointCalibrationDataset(CalibrationDataset):
    def __init__(self, img_folder, keys, width=None, height=None, centered=True, n_limit=None, mode='none', transform=None, target_transform=None):
        super().__init__(img_folder, keys, width, height, centered, None, mode, transform, target_transform)

        self.n_limit = n_limit
        self.selected_idxs = []
        for idx in range(len(self.keys)):
            _, labels = super().__getitem__(idx)
            pitch = Pitch.from_one_annotation(labels, return_none_if_bad_annotation=True)
            if pitch is not None:
                self.selected_idxs.append(idx)

            if self.n_limit is not None and len(self.selected_idxs) >= self.n_limit:
                break

    @classmethod
    def from_folder(cls, img_folder, width=None, height=None, centered=True, n_limit=None, mode='none', transform=None, target_transform=None):
        keys = [
            re.search(r'(\d+)\.jpg$', file).group(1) 
            for file in glob.glob(f'{img_folder}/*.jpg')
        ]
        return cls(img_folder, keys, width=width, height=height, centered=centered, n_limit=n_limit, mode=mode, transform=transform, target_transform=target_transform)


    def __len__(self):
        return len(self.selected_idxs)

    def __getitem__(self, idx):
        idx = self.selected_idxs[idx]
        image, labels = super().__getitem__(idx)
        pitch = Pitch.from_one_annotation(labels, return_none_if_bad_annotation=True)
        if pitch is None:
            return None, None
        return image, pitch.keypoints


class PitchHomographyCalibrationDataset(CalibrationDataset):
    def __init__(self, img_folder, keys, width=None, height=None, centered=True, n_limit=None, mode='none', transform=None, target_transform=None):
        super().__init__(img_folder, keys, width, height, centered, None, mode, transform, target_transform)

        self.n_limit = n_limit
        self.selected_idxs = []
        for idx in range(len(self.keys)):
            _, labels = super().__getitem__(idx)
            pitch = Pitch.from_one_annotation(labels, return_none_if_bad_annotation=True)
            if pitch is not None:
                self.selected_idxs.append(idx)

            if self.n_limit is not None and len(self.selected_idxs) >= self.n_limit:
                break

    @classmethod
    def from_folder(cls, img_folder, width=None, height=None, centered=True, n_limit=None, mode='none', transform=None, target_transform=None):
        keys = [
            re.search(r'(\d+)\.jpg$', file).group(1) 
            for file in glob.glob(f'{img_folder}/*.jpg')
        ]
        return cls(img_folder, keys, width=width, height=height, centered=centered, n_limit=n_limit, mode=mode, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.selected_idxs)

    def __getitem__(self, idx):
        idx = self.selected_idxs[idx]
        image, labels = super().__getitem__(idx)
        pitch = Pitch.from_one_annotation(labels, return_none_if_bad_annotation=True)
        if pitch is None:
            return None, None
        return image, pitch.homography