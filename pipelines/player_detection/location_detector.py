import torch
import numpy as np
from torchtyping import TensorType
from sklearn.mixture import BayesianGaussianMixture

import torchvision.transforms.functional as TorchvisionF
import pytorch_lightning as pl

class LocationDetector(pl.LightningModule):
    def __init__(self, max_components=30, max_iter=20, tol=0.01, num_samples=500, threshold=0.6):
        super().__init__()
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components, covariance_type='spherical',
            verbose=0,
            max_iter=max_iter
        )
        self.tol = tol
        self.num_samples = num_samples
        self.threshold = threshold

    def forward(self, heatmap: TensorType['batch', 'channel', 'row', 'column']) -> TensorType['n_components', 2]:
        """
        Args:
            heatmap (TensorType['row', 'column']): Heatmap, multi-channel
                + Note: only single-batched images are supported.

        Returns:
        """
        B, C, H, W = heatmap.shape              # pylint: disable=invalid-name
        if B != 1:                              # [TODO]: to support multibatch images
            raise NotImplementedError(f"Only supported for single-batched image. Received shape = {heatmap.shape}")
        if C != 1:
            heatmap = TorchvisionF.rgb_to_grayscale(heatmap)

        heatmap[heatmap < self.threshold] = 0
        heatmap_idxs = torch.nonzero(heatmap.view(H, W), as_tuple=False)

        if heatmap_idxs.shape[0] <= 0:
            return torch.tensor([]).view(0, 2)

        heatmap_vals = heatmap[0, 0, heatmap_idxs[:, 0], heatmap_idxs[:, 1]]
        probs = heatmap_vals / heatmap_vals.sum()

        chosen_idxs = torch.multinomial(probs, self.num_samples, replacement=True)
        heatmap_chosen_idxs = heatmap_idxs[chosen_idxs]

        bgmm = self.bgmm.fit(heatmap_chosen_idxs.cpu())
        chosen_idxs = np.where(bgmm.weights_ > self.tol)
        # print(bgmm.weights_)

        components = bgmm.means_[chosen_idxs] 
        return torch.tensor(components)