import torch
from torchtyping import TensorType
from tqdm import tqdm

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
            # y0, y1 = H_crop // 4, 3 * H_crop // 4
            # x0, x1 = W_crop // 4, 3 * W_crop // 4
            # cropped_output_pieces = mask_pieces[:, :, y0:y1, x0:x1]

            for mask_piece, offset_piece in zip(mask_pieces, offset_pieces):
                y0, x0, y1, x1 = offset_piece
                added_mask = mask_piece[0]

                # H_mask, W_mask V= added_mask.shape
                # added_mask = added_mask.reshape(H_mask * W_mask)
                # added_mask = F.softmax(added_mask, dim=-1)
                # added_mask = added_mask.reshape((H_mask, W_mask))

                output[0, :, y0:y1, x0:x1] += added_mask

        return output