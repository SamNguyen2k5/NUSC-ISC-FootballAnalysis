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

        outputs = [
            output_pieces + offset_pieces
            for output_pieces, offset_pieces in tqdm(self._for_each_piece_with_result(image))
        ]
        
        return torch.tensor(outputs)