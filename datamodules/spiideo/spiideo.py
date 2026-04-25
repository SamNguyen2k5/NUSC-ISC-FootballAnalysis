import re
import glob
import json
import warnings

from collections import defaultdict
from typing import Literal, List, Dict

import cv2
import numpy as np
from pydantic import BaseModel
from torch.utils.data import Dataset
from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs
from .coco import COCOModel, COCOImage, COCOAnnotation, COCOCategory

class SpiideoDatasetArgs(BaseModel):
    base_folder: str
    width: int
    height: int
    n_limit: int | None = None

class SpiideoBaseDataset(Dataset):
    @property
    def annotation_file_path(self) -> str:
        match self.mode:
            case 'valid':
                return f'{self.args.base_folder}/annotations/val.json'
            case _:
                return f'{self.args.base_folder}/annotations/{self.mode}.json'

    def image_path(self, file_name: str) -> str:
        match self.mode:
            case 'valid':
                return f'{self.args.base_folder}/val/{file_name}'
            case _:
                return f'{self.args.base_folder}/{self.mode}/{file_name}'

    def _load_coco_model(self) -> COCOModel:
        with open(self.annotation_file_path, 'r', encoding='utf-8') as f:
            coco_model = COCOModel(**json.load(f))

            images: Dict[int, COCOImage] = {}
            annotations: Dict[int, List[COCOAnnotation]] = defaultdict(list)
            categories: Dict[int, str] = defaultdict(str)
            
            original_width, original_height = 0, 0

            image: COCOImage
            for image in coco_model.images:
                if self.args.n_limit is not None and image.id >= self.args.n_limit:
                    break
                
                original_width, original_height = image.width, image.height
                images[image.id] = image
                images[image.id].width = self.args.width
                images[image.id].height = self.args.height

            annotation: COCOAnnotation
            flag = True

            for annotation in coco_model.annotations:
                if self.args.n_limit is not None and annotation.image_id >= self.args.n_limit:
                    break

                if flag:
                    print(annotation)

                scale_width = original_width // self.args.width
                scale_height = original_height // self.args.height 
                annotation.keypoints[:, 0] /= scale_width
                annotation.keypoints[:, 1] /= scale_height
                annotation.bbox //= np.array([scale_width, scale_height] * 2)

                if flag:
                    print(annotation)
                    flag = False

                annotations[annotation.image_id].append(annotation)

            category: COCOCategory
            for category in coco_model.categories:
                categories[category.id] = category.name

            return images, annotations, categories

    def __init__(self, 
                 args: SpiideoDatasetArgs, 
                 mode: Literal['mini', 'train', 'valid', 'test', 'challenge'], 
                 transform=None, 
                 target_transform=None):
        self.args = args
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.annotations, self.categories = self._load_coco_model()

    def __len__(self):
        if self.args.n_limit is None:
            return len(self.images)

        return self.args.n_limit

    def __getitem__(self, idx):
        raise NotImplementedError
        
class SpiideoDataset(SpiideoBaseDataset):
    def __getitem__(self, idx):
        # image_name = coco_model.images[idx]

        file_name = self.images[idx].file_name
        image = cv2.imread(self.image_path(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = cv2.resize(image, (self.args.width, self.args.height))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        return image

class SpiideoDataModule(BaseDataModule):
    def __init__(self, args_dataset: SpiideoDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['mini', 'train', 'valid', 'test', 'challenge']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either mini, train, valid, test or challenge.')

        return SpiideoDataset(self.args_dataset, mode=mode)
        
class SpiideoWithMetadataDataset(SpiideoBaseDataset):
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx].file_name
        image = cv2.imread(self.image_path(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = cv2.resize(image, (self.args.width, self.args.height))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        return image, self.images[idx], self.annotations[idx], self.categories[idx]

class SpiideoWithMetadataDataModule(BaseDataModule):
    def __init__(self, args_dataset: SpiideoDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['mini', 'train', 'valid', 'test', 'challenge']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either mini, train, valid, test or challenge.')

        return SpiideoWithMetadataDataset(self.args_dataset, mode=mode)