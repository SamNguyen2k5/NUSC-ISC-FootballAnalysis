import numpy as np
import torch
import torch.nn as nn
from .keypoint_mapping import PitchKeypointMapping
from .homography_solver import HomographySolver
from scripts.datasets.calibration import CalibrationLabelsMapping

KEYPOINT_DEFINITION = (
    (PitchKeypointMapping.CENTRE, None),
    (PitchKeypointMapping.CIRCLE_TEN_YARD_BOTTOM, None),
    (PitchKeypointMapping.CIRCLE_TEN_YARD_TOP, None),
    (PitchKeypointMapping.CIRCLE_TEN_YARD_LEFT, None),
    (PitchKeypointMapping.CIRCLE_TEN_YARD_RIGHT, None),
    (PitchKeypointMapping.FIELD_TOP_LEFT, 
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.SIDE_LINE_TOP)),
    (PitchKeypointMapping.FIELD_BOTTOM_LEFT,
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.SIDE_LINE_BOTTOM)),
    (PitchKeypointMapping.FIELD_TOP_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.SIDE_LINE_TOP)),
    (PitchKeypointMapping.FIELD_BOTTOM_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.SIDE_LINE_BOTTOM)),
    (PitchKeypointMapping.MIDFIELD_TOP,
        (CalibrationLabelsMapping.SIDE_LINE_TOP, CalibrationLabelsMapping.MIDDLE_LINE)),
    (PitchKeypointMapping.MIDFIELD_BOTTOM,
        (CalibrationLabelsMapping.SIDE_LINE_BOTTOM, CalibrationLabelsMapping.MIDDLE_LINE)),
    (PitchKeypointMapping.LEFT_PENALTY_TOP_LEFT,
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.BIG_RECT_LEFT_TOP)),
    (PitchKeypointMapping.LEFT_PENALTY_TOP_RIGHT,
        (CalibrationLabelsMapping.BIG_RECT_LEFT_MAIN, CalibrationLabelsMapping.BIG_RECT_LEFT_TOP)),
    (PitchKeypointMapping.LEFT_PENALTY_BOTTOM_LEFT,
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.BIG_RECT_LEFT_BOTTOM)),
    (PitchKeypointMapping.LEFT_PENALTY_BOTTOM_RIGHT,
        (CalibrationLabelsMapping.BIG_RECT_LEFT_MAIN, CalibrationLabelsMapping.BIG_RECT_LEFT_BOTTOM)),
    (PitchKeypointMapping.RIGHT_PENALTY_TOP_LEFT,
        (CalibrationLabelsMapping.BIG_RECT_RIGHT_MAIN, CalibrationLabelsMapping.BIG_RECT_RIGHT_TOP)),
    (PitchKeypointMapping.RIGHT_PENALTY_TOP_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.BIG_RECT_RIGHT_TOP)),
    (PitchKeypointMapping.RIGHT_PENALTY_BOTTOM_LEFT,
        (CalibrationLabelsMapping.BIG_RECT_RIGHT_MAIN, CalibrationLabelsMapping.BIG_RECT_RIGHT_BOTTOM)),
    (PitchKeypointMapping.RIGHT_PENALTY_BOTTOM_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.BIG_RECT_RIGHT_BOTTOM)),
    (PitchKeypointMapping.LEFT_GOALKEEPER_TOP_LEFT,
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.SMALL_RECT_LEFT_TOP)),
    (PitchKeypointMapping.LEFT_GOALKEEPER_TOP_RIGHT,
        (CalibrationLabelsMapping.SMALL_RECT_LEFT_MAIN, CalibrationLabelsMapping.SMALL_RECT_LEFT_TOP)),
    (PitchKeypointMapping.LEFT_GOALKEEPER_BOTTOM_LEFT,
        (CalibrationLabelsMapping.SIDE_LINE_LEFT, CalibrationLabelsMapping.SMALL_RECT_LEFT_BOTTOM)),
    (PitchKeypointMapping.LEFT_GOALKEEPER_BOTTOM_RIGHT,
        (CalibrationLabelsMapping.SMALL_RECT_LEFT_MAIN, CalibrationLabelsMapping.SMALL_RECT_LEFT_BOTTOM)),
    (PitchKeypointMapping.RIGHT_GOALKEEPER_TOP_LEFT,
        (CalibrationLabelsMapping.SMALL_RECT_RIGHT_MAIN, CalibrationLabelsMapping.SMALL_RECT_RIGHT_TOP)),
    (PitchKeypointMapping.RIGHT_GOALKEEPER_TOP_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.SMALL_RECT_RIGHT_TOP)),
    (PitchKeypointMapping.RIGHT_GOALKEEPER_BOTTOM_LEFT,
        (CalibrationLabelsMapping.SMALL_RECT_RIGHT_MAIN, CalibrationLabelsMapping.SMALL_RECT_RIGHT_BOTTOM)),
    (PitchKeypointMapping.RIGHT_GOALKEEPER_BOTTOM_RIGHT,
        (CalibrationLabelsMapping.SIDE_LINE_RIGHT, CalibrationLabelsMapping.SMALL_RECT_RIGHT_BOTTOM)),
)

def _after_mappings(labels, swap_coords=False):
    keypoint_idxs = []
    after_mapping = []

    for keypoint_idx, definition_idx in KEYPOINT_DEFINITION:
        if definition_idx is None:
            continue

        a, b = definition_idx
        line_a, line_b = labels[a], labels[b]

        # Either (poly-)lines not properly defined
        if line_a.shape[0] < 2 or line_b.shape[0] < 2:
            continue

        line_a = np.hstack((line_a[[0, -1], :], np.ones((2, 1))))
        line_b = np.hstack((line_b[[0, -1], :], np.ones((2, 1))))

        pt = np.cross(np.cross(line_a[0], line_a[1]), np.cross(line_b[0], line_b[1]))
        pt /= pt[2]

        if np.isnan(pt).any():
            continue

        # print(PitchKeypointMapping.forward(keypoint_idx))
        # print('\t', CalibrationLabelsMapping.forward(a), line_a, f' shape={line_a.shape}')
        # print('\t', CalibrationLabelsMapping.forward(b), line_b, f' shape={line_b.shape}')
        # print(f'\t => {pt}')

        keypoint_idxs.append(keypoint_idx)
        after_mapping.append(pt)
        
    if len(after_mapping) <= 1:
        return None, None
        
    X_COORD, Y_COORD, Z_COORD = 0, 1, 2
    if swap_coords:
        X_COORD, Y_COORD = Y_COORD, X_COORD

    after_mapping = np.array(after_mapping)
    after_mapping = torch.stack((
        torch.Tensor(after_mapping[:, X_COORD]),
        torch.Tensor(after_mapping[:, Y_COORD]),
        torch.Tensor(after_mapping[:, Z_COORD])
    ), dim=1)

    # print('After mapping:')
    # print(after_mapping)
    # print(after_mapping.shape)
    return keypoint_idxs, after_mapping

def _homography_matrix(old_pts, new_pts):
    return HomographySolver()(old_pts, new_pts)