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

class CalibrationLabelsMapping:
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

    NOT_FIELD = 0
    BIG_RECT_LEFT_BOTTOM = 1
    BIG_RECT_LEFT_MAIN = 2
    BIG_RECT_LEFT_TOP = 3
    BIG_RECT_RIGHT_BOTTOM = 4
    BIG_RECT_RIGHT_MAIN = 5
    BIG_RECT_RIGHT_TOP = 6
    CIRCLE_CENTRAL = 7
    CIRCLE_LEFT = 8
    CIRCLE_RIGHT = 9
    GOAL_LEFT_CROSSBAR = 10
    GOAL_LEFT_POST_LEFT  = 11
    GOAL_LEFT_POST_RIGHT = 12
    GOAL_RIGHT_CROSSBAR = 13
    GOAL_RIGHT_POST_LEFT = 14
    GOAL_RIGHT_POST_RIGHT = 15
    MIDDLE_LINE = 16
    SIDE_LINE_BOTTOM = 17
    SIDE_LINE_LEFT = 18
    SIDE_LINE_RIGHT = 19
    SIDE_LINE_TOP = 20
    SMALL_RECT_LEFT_BOTTOM = 21
    SMALL_RECT_LEFT_MAIN = 22
    SMALL_RECT_LEFT_TOP = 23
    SMALL_RECT_RIGHT_BOTTOM = 24
    SMALL_RECT_RIGHT_MAIN = 25
    SMALL_RECT_RIGHT_TOP = 26

    @classmethod
    def forward(cls, idx: int):
        return cls.CALIBRATION_LABELS[idx]

    @classmethod
    def backward(cls, label: str):
        return cls.CALIBRATION_LABELS_BACKWARD.get(label, None)

class CalibrationDataset(Dataset):
    img_folder: str
    raw_folder: str
    width: int
    height: int
    n_limit: int
    annotations_df: pd.DataFrame 
    keys: List[str]
    mode: str
    transform: any
    target_transform: any

    def __init__(self, img_folder, keys, width, height, centered=False, n_limit=None, mode='none', transform=None, target_transform=None):
        self.img_folder = img_folder
        self.width = width
        self.height = height
        self.n_limit = n_limit
        self.keys = keys
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.centered = centered

        matches_path = f'{self.img_folder}/match_info.json'
        with open(matches_path, "r") as f:
            self.match_infos = json.load(f)

    @classmethod
    def from_folder(cls, img_folder, width=None, height=None, centered=False, n_limit=None, mode='none', transform=None, target_transform=None):
        keys = [
            re.search(r'(\d+)\.jpg$', file).group(1) 
            for file in glob.glob(f'{img_folder}/*.jpg')
        ]
        return cls(img_folder, keys, width=width, height=height, centered=centered, n_limit=n_limit, mode=mode, transform=transform, target_transform=target_transform)

    def match_info(self, key):
        return self.match_infos.get(f'{key}.jpg')

    def __len__(self):
        if self.n_limit:
            return self.n_limit
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image_path = f'{self.img_folder}/{key}.jpg'
        label_path = f'{self.img_folder}/{key}.json'

        # Get image
        image = cv2.imread(image_path)
        image = image.astype('float32') / 255.0                     # Normalisation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # Colour scale conversion

        height, width = None, None
        if not self.width or not self.height:
            height, width, _ = image.shape
        else:
            image = cv2.resize(image, (self.width, self.height))
            height, width = self.height, self.width

        # Get calibration labels
        label = None
        with open(label_path, "r") as f:
            label = json.load(f)

        OFFSET_DX, OFFSET_DY = 0, 0
        if self.centered:
            OFFSET_DX, OFFSET_DY = -0.5, -0.5

        ordered_label = dict((k, np.array([])) for k in CalibrationLabelsMapping.ALL_LABELS)
        for name, coords in label.items():
            ordered_name = CalibrationLabelsMapping.backward(name)
            stacked_coords = np.vstack([
                np.array([int((coord['x'] + OFFSET_DX) * width), int((coord['y'] + OFFSET_DY) * height)]) 
                for coord in coords
            ])
            ordered_label[ordered_name] = stacked_coords

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        return image, ordered_label

if __name__ == '__main__':
    dataset = CalibrationDataset.from_folder('data/calibration/train', mode='train')
    print(dataset[0])