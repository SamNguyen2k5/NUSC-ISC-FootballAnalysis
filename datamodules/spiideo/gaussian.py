import torch
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
        heatmap += np.exp(exponent)
    
    return heatmap

class Spiideo2DGaussianMaskDataset(SpiideoWithMetadataDataset):
    def __getitem__(self, idx):
        metadata = self.images[idx]
        file_name = metadata.file_name

        # print(f'Reading from {self.image_path(file_name)}')

        image = cv2.imread(self.image_path(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = cv2.resize(image, (self.args.width, self.args.height))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ordered_label = self.target_transform(ordered_label)

        height, width = self.args.height, self.args.width

        centers = []
        sigmas = []

        for annotation in self.annotations[idx]:
            kps = []
            for x, y, z in annotation.keypoints.astype('int'):
                x0, y0 = int(x / z * width / metadata.width), int(y / z * height / metadata.height)
                kps.append((x0, y0))

            (x0, y0), (_, y1) = kps
            if y0 > y1:
                (_, y1), (x0, y0) = kps

            centers.append((x0, y0))
            sigmas.append((abs(y0 - y1), abs(y0 - y1)))

        gaussian_mask = _generate_2d_gmm_heatmap(
            shape=(height, width),
            centers=centers,
            sigmas=sigmas
        )

        # gaussian_mask = np.stack([gaussian_mask] * 3, axis=0)

        image = torch.tensor(image).unsqueeze(0)
        gaussian_mask = torch.tensor(gaussian_mask).unsqueeze(0)
        return image, gaussian_mask

class Spiideo2DGaussianMaskDataModule(BaseDataModule):
    def __init__(self, args_dataset: SpiideoDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['mini', 'train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either mini, train, valid or test.')

        return Spiideo2DGaussianMaskDataset(self.args_dataset, mode=mode)