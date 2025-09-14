import re
import os
import cv2
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from typing import List
from dataclasses import dataclass

class CalibrationLabels:
    """
    https://github.com/SoccerNet/sn-calibration > Soccer pitch annotations
    """
    CALIBRATION_LABELS: List[str] = [
        "Not field",
        "Big rect. left bottom",
        "Big rect. left main",
        "Big rect. left top",
        "Big rect. right bottom",
        "Big rect. right main",
        "Big rect. right top",
        "Circle central",
        "Circle left",
        "Circle right",
        "Goal left crossbar",
        "Goal left post left ",
        "Goal left post right",
        "Goal right crossbar",
        "Goal right post left",
        "Goal right post right",
        "Middle line",
        "Side line bottom",
        "Side line left",
        "Side line right",
        "Side line top",
        "Small rect. left bottom",
        "Small rect. left main",
        "Small rect. left top",
        "Small rect. right bottom",
        "Small rect. right main",
        "Small rect. right top"
    ]

    CALIBRATION_LABELS_BACKWARD = dict((v, k) for k, v in enumerate(CALIBRATION_LABELS))
    ALL_LABELS = range(len(CALIBRATION_LABELS))

    @classmethod
    def forward(cls, idx: int):
        return cls.CALIBRATION_LABELS[idx]

    @classmethod
    def backward(cls, label: str):
        return cls.CALIBRATION_LABELS_BACKWARD.get(label, None)

class CalibrationDataset(Dataset):
    img_folder: str
    raw_folder: str
    annotations_df: pd.DataFrame 
    keys: List[str]
    mode: str
    transform: any
    target_transform: any

    def __init__(self, img_folder, keys, mode='none', transform=None, target_transform=None):
        self.img_folder = img_folder
        self.keys = keys
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        matches_path = f'{self.img_folder}/match_info.json'
        with open(matches_path, "r") as f:
            self.match_infos = json.load(f)

    @classmethod
    def from_folder(cls, img_folder, mode, transform=None, target_transform=None):
        keys = [
            re.search(r'(\d+)\.jpg$', file).group(1) 
            for file in glob.glob(f'{img_folder}/*.jpg')
        ]
        return cls(img_folder, keys, mode=mode, transform=transform, target_transform=target_transform)

    def match_info(self, key):
        return self.match_infos.get(f'{key}.jpg')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image_path = f'{self.img_folder}/{key}.jpg'
        label_path = f'{self.img_folder}/{key}.json'

        # Get image
        image = cv2.imread(image_path)
        image = image.astype('float32') / 255.0                     # Normalisation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # Colour scale conversion
        height, width, _ = image.shape

        # Get calibration labels
        label = None
        with open(label_path, "r") as f:
            label = json.load(f)

        ordered_label = dict((k, []) for k in CalibrationLabels.ALL_LABELS)
        for name, coords in label.items():
            ordered_name = CalibrationLabels.backward(name)
            stacked_coords = np.vstack([np.array([int(coord['x'] * width), int(coord['y'] * height)]) for coord in coords])
            ordered_label[ordered_name] = stacked_coords

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        return image, ordered_label

if __name__ == '__main__':
    dataset = CalibrationDataset.from_folder('data/calibration/train', mode='train')
    print(dataset[0])