from typing import Optional
import torch
import numpy as np
import litserve as ls
from apis.payload import NumpyPayload

class FromImageAPIMixin:
    def decode_request(self, request: NumpyPayload) -> np.ndarray:
        return request.to_numpy()

class ToImageAPIMixin:
    def encode_response(self, output: torch.Tensor | np.ndarray | None) -> dict:
        match output:
            case torch.Tensor():
                return NumpyPayload.from_numpy(output.numpy()).model_dump()
            case np.ndarray():
                return NumpyPayload.from_numpy(output).model_dump()
            case None:
                return {}

class Img2ImgAPI(FromImageAPIMixin, ToImageAPIMixin, ls.LitAPI):
    def setup(self, device):
        self.device = device

    def predict(self, x: any) -> torch.Tensor:
        raise NotImplementedError