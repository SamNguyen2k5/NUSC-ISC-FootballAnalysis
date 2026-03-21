import torch
from torchtyping import TensorType

from pipelines.player_detection.cropper import Cropper

class CropThenStack(Cropper):
    def forward(self, image: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        raise NotImplementedError()

        H_crop, W_crop = self.window_crop   # pylint: disable=invalid-name
        output = torch.zeros_like(image)
        
        for output_pieces, offset_pieces in self._for_each_piece_with_result(image):
            # [TODO]: Add output coordinates with offset
            # Stack output coordinates into one big stacked Tensor
            output_pieces + offset_pieces

        return output