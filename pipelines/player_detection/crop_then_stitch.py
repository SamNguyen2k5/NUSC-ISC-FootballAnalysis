import torch
from torchtyping import TensorType

from pipelines.player_detection.cropper import Cropper

class CropThenStitch(Cropper):
    def forward(self, image: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        """
        Args:
            x (TensorType['batch', 'channel', 'row', 'column']): Image, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        H_crop, W_crop = self.window_crop   # pylint: disable=invalid-name
        output = torch.zeros_like(image)
        
        for mask_pieces, offset_pieces in self._for_each_piece_with_result(image):
            for mask_piece, offset_piece in zip(mask_pieces, offset_pieces):
                y0, x0 = offset_piece
                y1, x1 = y0 + H_crop, x0 + W_crop
                output[0, 0, y0:y1, x0:x1] += mask_piece[0]

        return output