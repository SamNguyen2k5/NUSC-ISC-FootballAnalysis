import torch, gc
from torch.nn import L1Loss

from importlib import resources
import wandb
import yaml

import gc
import cv2
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn.functional as TorchF
import torchvision.transforms.functional as TorchvisionF

import matplotlib.pyplot as plt

from datamodules.spiideo.coco import COCOPredictedAnnotation

from datamodules.spiideo import SpiideoWithMetadataDataModule, SpiideoDatasetArgs
from datamodules.base_datamodule import BaseDataModuleArgs
from datamodules.spiideo.gaussian import Spiideo2DGaussianMaskDataModule, Spiideo2DGaussianMaskDatasetArgs

from models.player_detection.unet_gaussian import UNetHeatmapGaussian
from pipelines.player_detection.location_detector import LocationDetector
from pipelines.player_detection.player_detector import PlayerDetector

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_args = SpiideoDatasetArgs(**config['spiideo_dataset'])
spiideo_data_module = SpiideoWithMetadataDataModule(
    SpiideoDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule'])
)

DEVICE = 'mps'
DTYPE = torch.float32
# Loading checkpoint error stuff
torch.hub.load('sm00thix/unet', 'unet', pretrained=False)

full_model = UNetHeatmapGaussian \
    .load_from_checkpoint(config['unet_with_gaussian_layer']) \
    .to(device=DEVICE, dtype=DTYPE) \
    .eval()

gaussian_model = full_model.heatmap_unet \
    .to(device=DEVICE, dtype=DTYPE) \
    .eval()

pitch_cover_model = full_model.masking_unet \
    .to(device=DEVICE, dtype=DTYPE) \
    .eval()

full_model = UNetHeatmapGaussian(
    pitch_cover_model, gaussian_model, lmbd_gaussianiser=1, 
    max_sigma=4, top_k=10000, select_k=1000
) \
    .to(device=DEVICE, dtype=DTYPE) \
    .eval()

location_detector = LocationDetector(
    tol=0.001, num_samples=10000
) \
    .to(device=DEVICE, dtype=DTYPE) \
    .eval()

player_detector = PlayerDetector(
    full_model,
    pitch_cover_model,
    location_detector,
    keypoint_threshold=0.99,
)

if __name__ == '__main__':
    TASK = 'challenge'
    ds = spiideo_data_module.dataset(TASK)
    print(f'Dataset loaded with length {len(ds)}.')

    for image_idx in tqdm(range(len(ds))):
    # for image_idx in (49,):
        gc.collect()
        torch.mps.empty_cache()

        image_id_repr = str(image_idx).zfill(6)

        try:
            img, coco_img, _, *_ = ds[image_idx]
            img_tensor = torch.tensor(img) \
                .permute((2, 0, 1)) \
                .unsqueeze(0) \
                .to(DEVICE, DTYPE)

            keypoints = player_detector(img_tensor)
            keypoints = keypoints.numpy()

            coords = coco_img.coordinates_from_image_to_world(keypoints[:, 1], keypoints[:, 0])
            print(f'Image #{image_id_repr}: Predicted {coords.shape[1]} keypoints')

            predicted_annotations = []
            for idx in range(coords.shape[1]):
                x, y = coords[:, idx]
                annot = COCOPredictedAnnotation(
                    id=idx,
                    image_id=image_idx,
                    category_id=1,
                    position_on_pitch=[x, y, 0]
                )

                predicted_annotations.append(annot.model_dump())

            with open(f'experiments/ex07_inference/results-{TASK}/{image_id_repr}.json', 'w', encoding='utf-8') as f:
                json.dump(predicted_annotations, f)

        except Exception as e:
            print('== Exception at image_idx={image_idx}')
            print(e)
            continue
