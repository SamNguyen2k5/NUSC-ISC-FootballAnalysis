import torch
from tqdm import tqdm
from torchtyping import TensorType

from pipelines.player_detection.cropper import Cropper

class CropThenStack(Cropper):
    def forward(self, image: TensorType['batch', 'channel', 'row', 'column']):
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """

        outputs = []
        offsets = []
        for output, offset in tqdm(self._for_each_piece_with_result(image)):
            outputs.append(output)
            offsets.append(offset.repeat(len(output), 1))
        
        return torch.concatenate(outputs), torch.concatenate(offsets)