import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as TorchVisionF
import torch.nn.functional as TorchF
from typing import Literal, List, Dict

import matplotlib.pyplot as plt

import cv2
import numpy as np
from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs
from .spiideo import SpiideoDatasetArgs, SpiideoWithMetadataDataset

def _generate_2d_gmm_heatmap(shape, centers, sigmas):
    """
    Generates a 2D Gaussian Mixture Map.
    
    Args:
        shape: Tuple (height, width) for the output map.
        centers: List of tuples [(x1, y1), (x2, y2), ...]
        sigmas: List of sigmas. Can be a single value [s1, s2] 
                or tuples for anisotropic shapes [(sx1, sy1), ...].
    """
    h, w = shape
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Initialize an empty heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for (cx, cy), s in zip(centers, sigmas):
        # Handle both circular (scalar) and elliptical (tuple) sigmas
        sx, sy = (s, s) if isinstance(s, (int, float)) else s
        
        # Calculate the 2D Gaussian exponent
        exponent = -((x_grid - cx)**2 / (2 * sx**2) + (y_grid - cy)**2 / (2 * sy**2))
        
        # Add the Gaussian to the mixture
        blob = np.exp(exponent).astype(np.float32)
        blob /= blob.max()
        heatmap += blob

    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

class Spiideo2DGaussianMaskDatasetArgs(SpiideoDatasetArgs):
    sigma: int = 16
    positive_sample_rate: float = 0.95
    oversampling_rate: float = 5.
    n_crops: int
    crop_width: int
    crop_height: int
    crop_export_width: int
    crop_export_height: int
    in_channels: int = 1

    @property
    def n_positive_samples(self):
        return int(self.n_crops * self.positive_sample_rate)

    @property
    def n_negative_samples(self):
        return self.n_crops - int(self.n_crops * self.positive_sample_rate)

    @property
    def n_oversamples(self):
        return int(self.oversampling_rate * self.n_crops)

class Spiideo2DGaussianMaskDataset(SpiideoWithMetadataDataset):
    def __init__(self, 
                 args: Spiideo2DGaussianMaskDatasetArgs, 
                 mode: Literal['mini', 'train', 'valid', 'test'], 
                 transform=None, 
                 target_transform=None):
        super().__init__(args, mode, transform, target_transform)
        self.args = args
        self.cropper = v2.RandomCrop(size=(self.args.crop_width, self.args.crop_height))

        self.current_image_idx = None
        self.current_images: None | List = None
        self.current_masks: None | List = None

    def __len__(self):
        return len(self.images) * self.args.n_crops

    def __getitem__(self, idx):
        image_idx = idx // self.args.n_crops
        crop_idx = idx % self.args.n_crops

        # [TODO]: check if crops are appropriately distributed. 
        # too many negative samples for now.

        if image_idx == self.current_image_idx:
            return self.current_images[crop_idx], self.current_masks[crop_idx]

        metadata = self.images[image_idx]
        file_name = metadata.file_name

        # print(f'Reading from {self.image_path(file_name)}')

        image = cv2.imread(self.image_path(file_name))
        image = cv2.resize(image, (self.args.width, self.args.height))
        image = image.astype('float32') / 255.0

        if self.args.in_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = torch.tensor(image).unsqueeze(0)
        elif self.args.in_channels == 3:
            image = torch.tensor(image).permute((2, 0, 1))
        else:
            raise NotImplementedError("Function not supported for images with {self.args.in_channels} colour channels")

        height, width = self.args.height, self.args.width

        centers = []
        sigmas = []

        for annotation in self.annotations[image_idx]:
            kps = []
            for x, y, z in annotation.keypoints.astype('int'):
                x0, y0 = int(x / z * width / metadata.width), int(y / z * height / metadata.height)
                kps.append((x0, y0))

            (x0, y0), (_, y1) = kps
            if y0 < y1:
                (_, y1), (x0, y0) = kps

            centers.append((x0, y0))
            # sigmas.append((abs(y0 - y1), abs(y0 - y1)))
            sigmas.append(self.args.sigma)

        gaussian_mask = _generate_2d_gmm_heatmap(
            shape=(self.args.height, self.args.width),
            centers=centers,
            sigmas=sigmas
        )

        gaussian_mask = torch.tensor(gaussian_mask).unsqueeze(0)

        selections = []
        for _ in range(self.args.n_oversamples):
            cropper_params = self.cropper.get_params(
                img=image, 
                output_size=(self.args.crop_height, self.args.crop_width)
            )

            cropped_image = TorchVisionF.crop(image, *cropper_params)
            cropped_gaussian_mask = TorchVisionF.crop(gaussian_mask, *cropper_params)
            selections.append((cropped_image, cropped_gaussian_mask, cropped_gaussian_mask.max()))

        selections = sorted(selections, key=lambda x: x[-1], reverse=True)
        selections = selections[:self.args.n_positive_samples] + selections[-self.args.n_negative_samples:]

        images, gaussian_masks, _ = zip(*selections)
        images = torch.stack(images)
        gaussian_masks = torch.stack(gaussian_masks)

        images = TorchF.interpolate(images, 
            size=(self.args.crop_export_height, self.args.crop_export_width), 
            mode='bilinear', align_corners=False)

        gaussian_masks = TorchF.interpolate(gaussian_masks, 
            size=(self.args.crop_export_height, self.args.crop_export_width), 
            mode='bilinear', align_corners=False)

        # gaussian_masks = gaussian_masks.view(-1, 1, self.args.crop_export_height * self.args.crop_export_width)
        # gaussian_masks = TorchF.softmax(gaussian_masks, dim=-1)
        # gaussian_masks = gaussian_masks.view(-1, 1, self.args.crop_export_height, self.args.crop_export_width)

        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            gaussian_masks = self.target_transform(gaussian_masks)

        self.current_image_idx = image_idx
        self.current_images, self.current_masks = images, gaussian_masks
        return images[crop_idx], gaussian_masks[crop_idx]

class Spiideo2DGaussianMaskDataModule(BaseDataModule):
    def __init__(self, args_dataset: SpiideoDatasetArgs, args_datamodule: BaseDataModuleArgs, debug=False):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset
        self.debug = debug

    def dataset(self, mode: str):
        if mode not in ['mini', 'train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either mini, train, valid or test.')

        if self.debug and mode == 'train':
            mode = 'mini'

        return Spiideo2DGaussianMaskDataset(
            self.args_dataset, mode=mode
        )