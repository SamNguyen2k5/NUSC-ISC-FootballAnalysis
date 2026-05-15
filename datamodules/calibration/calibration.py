import re
import glob
import json
import warnings
from itertools import islice
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel
from torch.utils.data import Dataset

from dataobjects.pitch import CalibrationLabelsMapping
from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs

class CalibrationDatasetArgs(BaseModel):
    """
    Args:
        img_folder (str): Image folder, consisting of jpg and json pairs.
        width (int): Width of the image, in px
        height (int): Height of the image, in px

    Kwargs:
        mode (str): 'train'/'valid'/'test'. Default: 'none'
        transform (Callable): transformation of the input. Default: None
        target_transform (Callable): transformation of the output. Default: None
        centered (bool): 
            - If True, coordinates will be in the range [-W/2, W/2] and [-H/2, H/2].
            - If False, coordinates will be in the range [0, W] and [0, H].
    """        
    img_folder: str
    width: int
    height: int

    n_limit: int = None
    centered: bool = False

class CalibrationDataset(Dataset):
    def __init__(self, args: CalibrationDatasetArgs, mode: str, transform=None, target_transform=None):
        self.img_folder = f'{args.img_folder}/{mode}'
        self.width = args.width
        self.height = args.height
        self.n_limit = args.n_limit

        img_url_iterator = glob.glob(f'{self.img_folder}/*.jpg')
        if self.n_limit is not None:
            img_url_iterator = islice(img_url_iterator, self.n_limit)

        self.keys = [re.search(r'(\d+)\.jpg$', file).group(1) for file in img_url_iterator]
        self.centered = args.centered
        matches_path = f'{self.img_folder}/match_info.json'
        with open(matches_path, "r", encoding='utf-8') as f:
            self.match_infos = json.load(f)

        self.transform = transform
        self.target_transform = target_transform

        # Debugging
        print(f'{len(self.keys)} JPEG images imported')

    @classmethod
    def from_folder(cls, **kwargs):
        warnings.warn('Depercated! Use the default constructor instead.')

    def match_info(self, key):
        return self.match_infos.get(f'{key}.jpg')

    def __len__(self):
        if self.n_limit:
            return self.n_limit
        return len(self.keys)
    
    def _raw_image_label(self, idx):
        key = self.keys[idx]
        image_path = f'{self.img_folder}/{key}.jpg'
        label_path = f'{self.img_folder}/{key}.json'

        image = cv2.imread(image_path)                              # Get image
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
        with open(label_path, "r", encoding='utf-8') as f:
            label = json.load(f)

        return image, label, width, height

    def __getitem__(self, idx):
        image, label, width, height = self._raw_image_label(idx=idx)

        offset_dx, offset_dy = 0, 0
        if self.centered:
            offset_dx, offset_dy = -0.5, -0.5

        ordered_label = dict((k, np.array([])) for k in CalibrationLabelsMapping.ALL_LABELS)
        for name, coords in label.items():
            ordered_name = CalibrationLabelsMapping.backward(name)
            stacked_coords = np.vstack([
                np.array([int((coord['x'] + offset_dx) * width), int((coord['y'] + offset_dy) * height)])
                for coord in coords
            ])
            ordered_label[ordered_name] = stacked_coords

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        return image, ordered_label

class CalibrationDataModule(BaseDataModule):
    def __init__(self, args_dataset: CalibrationDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either train, valid or test.')

        return CalibrationDataset(self.args_dataset, mode=mode)