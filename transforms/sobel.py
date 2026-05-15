import torch
import torch.nn.functional as F
from torchtyping import TensorType

GX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
GY = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

def sobel_magnitude(batch_tensor: TensorType['batch', 1, 'row', 'column']) -> TensorType['batch', 1, 'row', 'column']:
    gx = GX.to(batch_tensor.device, batch_tensor.dtype)
    gy = GY.to(batch_tensor.device, batch_tensor.dtype)
    grad_x = F.conv2d(batch_tensor, gx, padding=1)              # pylint: disable=not-callable
    grad_y = F.conv2d(batch_tensor, gy, padding=1)              # pylint: disable=not-callable
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    return magnitude