import base64
import yaml
from pathlib import Path
import numpy as np
import torch
import litserve as ls
from models.field_registration import Segmentation

from apis.payload import NumpyPayload
from apis.inference.base.image import Img2ImgAPI

with Path(__file__).parent.joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)
    print(config)

class SegmentationInferenceAPI(Img2ImgAPI):
    def setup(self, device):
        super().setup(device)
        self.model = Segmentation.load_from_checkpoint(config['checkpoint']['segmentation'], weights_only=True) \
                                 .to(device).float()
        self.model.eval()

    def predict(self, x: any) -> torch.Tensor:
        batched_x = torch.tensor(x).unsqueeze(0).unsqueeze(0) \
            .to(self.device).float()

        with torch.no_grad():
            mask = self.model.forward(batched_x)
            mask = mask.detach().cpu()
            mask = mask[0, 0, :]
            return mask