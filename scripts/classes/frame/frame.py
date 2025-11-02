import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

class FrameNumpy():
    def __init__(self, points, segments):
        self.points_ = points
        self.segments = segments

    @property
    def points(self):
        self.points_ /= self.points_[:, [2]]
        return self.points_

    def apply_homography(self, H: torch.Tensor):
        # Need (H * P^T)^T = P * H^T
        self.points_ = self.points_ @ H.T
        return self

    def plot(self, ax=None, scale=1, thickness=None, extra_space_scale=1, has_axes=True, color_seg='red', color_pts='red', alpha=1):
        if not thickness:
            thickness = (scale + 1) // 2

        X = np.stack([
            self.points[self.segments[:, 0].astype('int'), 0:2],
            self.points[self.segments[:, 1].astype('int'), 0:2]
        ], axis=1)
        ax.plot(X[:, :, 1].T, X[:, :, 0].T, marker='o', color=color_seg, linewidth=3, alpha=alpha)
        ax.scatter(x=self.points[:, 1], y=self.points[:, 0], color=color_pts, marker='v', s=50, alpha=alpha)