import torch
from torchtyping import TensorType
from sklearn.mixture import BayesianGaussianMixture
import pytorch_lightning as pl

class LocationDetector(pl.LightningModule):
    def __init__(self, max_components=20):
        super().__init__()
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components, covariance_type='spherical', verbose=1
        )

    def forward(self, heatmap: TensorType['row', 'column']) -> TensorType['n_components', 2]:
        """
        Args:
            heatmap (TensorType['row', 'column']): Heatmap, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        heatmap_idxs = torch.nonzero(heatmap, as_tuple=False)
        components = self.bgmm.fit(heatmap_idxs).means_
        return torch.tensor(components)