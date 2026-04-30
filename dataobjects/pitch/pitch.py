import json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import cv2

from dataobjects.pitch.keypoint_mapping import PitchKeypointMapping
from dataobjects.pitch.helper import _after_mappings, _homography_matrix
from dataobjects.frame import FrameNumpy

class Pitch(nn.Module):
    N_KEYPOINTS = len(PitchKeypointMapping.PITCH_LABELS)

    POLYGONS = (
        PitchKeypointMapping.BOX_FIELD_LEFT_HALF,
        PitchKeypointMapping.BOX_FIELD_RIGHT_HALF,
        PitchKeypointMapping.BOX_PENALTY_LEFT,
        PitchKeypointMapping.BOX_PENALTY_RIGHT,
        PitchKeypointMapping.BOX_GOALKEEPER_LEFT,
        PitchKeypointMapping.BOX_GOALKEEPER_RIGHT,
    )

    POLYGONS_FIELD = (
        PitchKeypointMapping.BOX_FIELD_LEFT_HALF,
        PitchKeypointMapping.BOX_FIELD_RIGHT_HALF,
    )

    ORIGINAL_HEIGHT = 75
    ORIGINAL_WIDTH = 110

    ELLIPSES = (
        (PitchKeypointMapping.ELLIPSES_TEN_YARD, True),
    )

    @staticmethod
    def LINE_SEGMENTS():
        segments = []
        for pts in Pitch.POLYGONS:
            pts_shift = pts[1:] + pts[:1]
            for x, y in zip(pts, pts_shift):
                segments.append([x, y])

        return segments

    def __init__(self, height=None, width=None, pitch_data=None):
        super().__init__()

        if height is None:
            height = 180
        if width is None:
            width = 320
        if pitch_data is None:
            pitch_data = 'templates/football_pitch.json'

        if isinstance(pitch_data, str):
            with open(pitch_data) as f:
                data_dict = json.load(f)
        else:
            data_dict = pitch_data

        self.width = width
        self.height = height
        self.keypoints = torch.cat((
            torch.Tensor(list(data_dict.values())), 
            torch.ones(Pitch.N_KEYPOINTS).unsqueeze(-1)
        ), dim=1)
        self.keypoints = self.keypoints.float()
        # self.keypoints = nn.Parameter(self.keypoints)
        self.homography = torch.eye(3)

    @classmethod
    def from_one_annotation(cls, labels, height=None, width=None, return_none_if_bad_annotation=False):
        pitch = cls(height=height, width=width)
        idxs, after_mappings = _after_mappings(labels, swap_coords=True)

        # Not enough points for the four-point algorithm
        if idxs is None or len(idxs) < 4:
            return None if return_none_if_bad_annotation else pitch

        H = pitch.forward(idxs, after_mappings)[0]
        return pitch.apply_homography(H)
    
    def batched_keypoints(self, n_batch=1, idxs=None):
        keypoints = self.keypoints
        # print(f'Device = {keypoints.device}')
        if idxs is not None:
            keypoints = keypoints[idxs, :]
        return torch.stack([keypoints] * n_batch, dim=0) 

    # @staticmethod
    # def all_keypoints_from_four_box_field_points(pts, height=None, width=None):
    #     n_batch = pts.shape[0]
    #     pitch = Pitch(height=height, width=width)

    #     pitch_box_field_keypoints = pitch.keypoints[list(PitchKeypointMapping.BOX_FIELD)]
    #     before_mappings = torch.stack([pitch_box_field_keypoints] * n_batch, dim=0)         # n_batch x 4 x 3
    #     H = (before_mappings, pts)                                        # n_batch x 3 x 3 

    #     all_keypoints = torch.stack([pitch.keypoints] * n_batch, dim=0)                     # n_batch x n_points x 3
    #     return torch.matmul(all_keypoints, H.transpose(2, 1))                # Batch multiplication

    def apply_homography(self, H: torch.Tensor, return_deepcopy=True) -> "Pitch":
        # Need (H * P^T)^T = P * H^T
        # print(self.keypoints, H.T)
        # print(self.keypoints.shape, H.T.shape)
        instance = self if not return_deepcopy else deepcopy(self)

        instance.keypoints = nn.Parameter(torch.matmul(instance.keypoints, H.T))
        instance.homography = torch.matmul(instance.homography, H)
        return instance

    # Homography mapping from current field to target field (given keypoints in the target field)
    def forward(self, idxs, after_mappings):
        before_mappings = self.keypoints[idxs].unsqueeze(0)
        after_mappings = after_mappings.unsqueeze(0)
        return _homography_matrix(before_mappings, after_mappings)

    def to_frame_numpy(self):
        points = self.keypoints.detach().numpy()
        segments = []
        # for polygon_labels in Pitch.POLYGONS:
        #     polygon_labels_shift = polygon_labels[1:] + polygon_labels[:1]
        #     for x, y in zip(polygon_labels, polygon_labels_shift):

        for x, y in PitchKeypointMapping.EDGES:
            segments.append([x, y, 1])

        segments = np.array(segments)
        return FrameNumpy(points, segments)


    def plot(self, ax=None, scale=1, thickness=None, extra_space_scale=1, stroke_intensity=255, focus_frame=False, has_axes=True):
        if not thickness:
            thickness = (scale + 1) // 2

        # Retain normalised x and y.
        # print(self.keypoints, self.keypoints.shape)

        height = self.height * scale * extra_space_scale
        width = self.width  * scale * extra_space_scale
        SHIFT = np.array([(width - self.width) // 2, (height - self.height) // 2])

        X = self.keypoints / self.keypoints[:, 2].unsqueeze(-1)
        X = X[:, [1, 0]] * scale
        X = X.detach().numpy()

        # Polygons / Lines
        img = np.zeros((height, width), dtype='int32')

        if focus_frame:
            img = cv2.rectangle(    
                img, 
                ((width - self.width * scale) // 2, (height - self.height * scale) // 2),
                ((width + self.width * scale) // 2, (height + self.height * scale) // 2),
                color=125, thickness=2 * thickness)

        polys = [X[list(polygon_labels)] for polygon_labels in Pitch.POLYGONS]
        polys = np.array(polys, dtype='int32')
        polys += SHIFT
        img = cv2.polylines(img, polys, True, stroke_intensity, thickness)

        # for polygon_labels in Pitch.POLYGONS:
        #     print([PitchKeypointMapping.forward(lbl) for lbl in polygon_labels])
        #     print(X[list(polygon_labels)])
        #     print(X[list(polygon_labels)].astype('int'))

        # Conics
        # for ((centre, top, bottom, left, right), _has_axes) in Pitch.ELLIPSES:
        #     centre_pt = X[centre]
        #     centre_pt = centre_pt.astype('int32')
        #     # print(np.linalg.norm(self._xy_point(top) - self._xy_point(bottom)))

        #     left_pt = X[left]
        #     right_pt = X[right]
        #     top_pt = X[top]
        #     bottom_pt = X[bottom]

        #     major_axis = int(np.linalg.norm(top_pt - bottom_pt) / 2)
        #     minor_axis = int(np.linalg.norm(left_pt - right_pt) / 2)

        #     major = left_pt - right_pt
        #     alpha = np.arctan2(major[0], major[1]) / np.pi * 180
        #     # print(alpha, type(alpha))

        #     img = cv2.ellipse(img, centre_pt, (major_axis, minor_axis), -alpha, 0, 360, color=255, thickness=thickness)
        #     # print(centre_pt, major_axis, minor_axis)

        #     bottom_pt = bottom_pt.astype('int32')
        #     right_pt = right_pt.astype('int32')

        #     # Annotations
        #     if _has_axes and has_axes:
        #         img = cv2.arrowedLine(img, centre_pt, bottom_pt, 125, thickness, tipLength=0.15)
        #         img = cv2.arrowedLine(img, centre_pt, right_pt, 125, thickness, tipLength=0.15)

        if ax:
            ax.imshow(img, cmap='gray', 
                extent=[
                    (-extra_space_scale + 1) * self.width // 2, 
                    (extra_space_scale + 1) * self.width // 2, 
                    (extra_space_scale + 1) * self.height // 2, 
                    (-extra_space_scale + 1) * self.height // 2
                ]
            )

        return img
    
    def plot_filled(self, ax=None, scale=1, thickness=None, extra_space_scale=1, has_axes=True):
        if not thickness:
            thickness = (scale + 1) // 2

        height = self.height * scale * extra_space_scale
        width = self.width * scale * extra_space_scale
        SHIFT = np.array([(width - self.width) // 2, (height - self.height) // 2])

        min_x = (-extra_space_scale + 1) * self.width // 2
        max_x = (extra_space_scale + 1) * self.width // 2
        min_y = (-extra_space_scale + 1) * self.height // 2
        max_y = (extra_space_scale + 1) * self.height // 2

        X = self.keypoints / self.keypoints[:, 2].unsqueeze(-1)
        X = X[:, [1, 0]] * scale
        X = X.detach().numpy()

        # Polygons / Lines
        inv_H = torch.inverse(self.homography).numpy()

        y0, x0 = np.meshgrid(
            np.arange(self.height),
            np.arange(self.width),
            indexing='ij'
        )

        pts_0 = np.stack([y0, x0, np.ones_like(y0)], axis=-1).reshape(-1, 3)
        pts_1 = pts_0 @ inv_H.T
        y1, x1 = (pts_1[:, [0, 1]] / pts_1[:, [2]]).T

        is_infield = np.logical_and(
            abs(y1) <= self.ORIGINAL_HEIGHT / 2,
            abs(x1) <= self.ORIGINAL_WIDTH / 2
        )

        img = is_infield.reshape((self.height, self.width)).astype('float32')

        if ax:
            ax.imshow(img, cmap='gray', 
                extent=[min_x, max_x, min_y, max_y],
                vmin=0,
                vmax=1
            )

        return img