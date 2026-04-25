import torch
import torch.nn.functional as TorchF
import torchvision.transforms.functional as TorchVisionF
from torchtyping import TensorType
import pytorch_lightning as pl

class HeatmapGaussianiser(pl.LightningModule):
    def __init__(self, threshold=0.0, top_k=1000, select_k=50, max_sigma=8):
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.select_k = select_k
        self.max_sigma = max_sigma

    def _generate_batched_gmm_heatmap(self, shape, centers, sigmas):
        """
        Args:
            shape: (H, W)
            centers: [B, N, 2] -> (x, y)
            sigmas: [B, N, 2] -> (sx, sy)
        Returns:
            heatmap: [B, H, W]
        """
        B, N, _ = centers.shape
        H, W = shape
        device = centers.device

        # 1. Create coordinate grids [H, W]
        y_range = torch.arange(H, dtype=torch.float32, device=device)
        x_range = torch.arange(W, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

        # 2. Reshape grids for broadcasting: [1, 1, H, W]
        x_grid = x_grid.view(1, 1, H, W)
        y_grid = y_grid.view(1, 1, H, W)

        # 3. Reshape centers and sigmas: [B, N, 1, 1]
        cx = centers[:, :, 1].view(B, N, 1, 1)
        cy = centers[:, :, 0].view(B, N, 1, 1)
        sx = sigmas[:, :, 0].view(B, N, 1, 1)
        sy = sigmas[:, :, 1].view(B, N, 1, 1)

        # 4. Compute exponent: [B, N, H, W]
        # Equation: -((x-cx)^2 / (2*sx^2) + (y-cy)^2 / (2*sy^2))
        eps = 1e-8
        exponent = -( ((x_grid - cx)**2 / (2 * sx**2 + eps)) + 
                    ((y_grid - cy)**2 / (2 * sy**2 + eps)) )
        
        # 5. Generate Blobs
        blobs = torch.exp(exponent)

        # 6. Normalize each blob to max 1: [B, N, H, W]
        # We find max over the (H, W) dimensions for each individual blob
        blob_max = blobs.view(B, N, -1).max(dim=-1)[0].view(B, N, 1, 1)
        blobs = blobs / (blob_max + eps)

        # 7. Sum across N (number of components) and clamp
        # Summing turns [B, N, H, W] -> [B, H, W]
        heatmap = torch.sum(blobs, dim=1)
        return torch.clamp(heatmap, 0, 1).view(B, 1, H, W)

    def forward(self, heatmap: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['batch', 'channel', 'row', 'column']:
        B, C, H, W = heatmap.shape          # pylint: disable=invalid-name
        if C == 3:
            heatmap = TorchVisionF.rgb_to_grayscale(heatmap)

        sigmas = heatmap.view(B, -1)
        sigmas = TorchF.softmax(sigmas, dim=-1)

        sigma_top, idxs = torch.topk(sigmas, k=self.top_k, dim=1)
        D = self.top_k // self.select_k     # pylint: disable=invalid-name
        sigma_top, idxs = sigma_top[:, ::D], idxs[:, ::D]

        ys, xs = torch.unravel_index(idxs, (H, W))
        centers = torch.stack([ys, xs], dim=-1)
        sigmas = sigma_top.unsqueeze(-1).expand(B, -1, 2)
        sigmas = self.max_sigma * sigmas

        return self._generate_batched_gmm_heatmap((H, W), centers, sigmas)

