from pathlib import Path
import yaml
import base64
import numpy as np
import torch
import litserve as ls
from models.field_registration import Segmentation
from apis.payload import NumpyPayload

with Path(__file__).parent.joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)
    print(config)

class SegmentationInference(ls.LitAPI):
    def setup(self, device):
        self.model = Segmentation.load_from_checkpoint(config['checkpoint']['segmentation'], weights_only=True) \
                                 .to(device).float()
        self.model.eval()
        self.device = device

    def decode_request(self, request: NumpyPayload) -> np.ndarray:
        decoded_bytes = base64.b64decode(request.encoded_bytes)
        img = np.frombuffer(decoded_bytes, dtype=request.dtype)
        img = np.reshape(img, request.dimensions)
        return img

    def predict(self, x: any):
        batched_x = torch.tensor(x).unsqueeze(0).unsqueeze(0) \
            .to(self.device).float()

        with torch.no_grad():
            mask = self.model.forward(batched_x)
            return mask.detach().cpu()

    def encode_response(self, output: torch.Tensor) -> dict:
        img = output[0, 0, :, :].numpy()

        print(len(base64.b64encode(img.tobytes()).decode()))
        print(tuple(img.shape))
        print(str(img.dtype))

        response = NumpyPayload(
            encoded_bytes=base64.b64encode(img.tobytes()).decode(),
            dimensions=tuple(img.shape),
            dtype=str(img.dtype)
        )
        return response.model_dump()