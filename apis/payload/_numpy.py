import base64
from typing import List

import numpy as np
from pydantic import BaseModel

class NumpyPayload(BaseModel):
    encoded_bytes: str
    dimensions: List[int]
    dtype: str

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> "NumpyPayload":
        return NumpyPayload(
            encoded_bytes=base64.b64encode(x.tobytes()).decode(),
            dimensions=list(x.shape),
            dtype=str(x.dtype)
        )

    def to_numpy(self) -> np.ndarray:
        decoded_bytes = base64.b64decode(self.encoded_bytes)
        x = np.frombuffer(decoded_bytes, dtype=self.dtype)
        x = np.reshape(x, self.dimensions)
        return x