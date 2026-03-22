import torch
from torchtyping import TensorType
from sklearn.mixture import BayesianGaussianMixture
import pytorch_lightning as pl

class LocationDetector(pl.LightningModule):
    def __init__(self, max_components=20, max_iter=20):
        super().__init__()
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components, covariance_type='spherical',
            verbose=0,
            max_iter=max_iter
        )

    def forward(self, heatmap: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['n_components', 2]:
        """
        Args:
            heatmap (TensorType['row', 'column']): Heatmap, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        B, C, H, W = heatmap.shape              # pylint: disable=invalid-name
        if B != 1:
            raise NotImplementedError(f"Only supported for single-batched image. Received shape = {heatmap.shape}")
        if C != 1:
            raise NotImplementedError(f"Only black and white images. Received shape: {heatmap.shape}")

        heatmap_idxs = torch.nonzero(heatmap.view(H, W), as_tuple=False)
        components = self.bgmm.fit(heatmap_idxs.cpu()).means_
        return torch.tensor(components)