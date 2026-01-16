from typing import List

import numpy as np
import torch

from .keypoint_mapping import PitchKeypointMapping
from .homography_solver import HomographySolver

class CalibrationLabelsMapping:
    """
    https://github.com/SoccerNet/sn-calibration > Soccer pitch annotations
    """
    CALIBRATION_LABELS: List[str] = [
        "Not field",
        "Big rect. left bottom",
        "Big rect. left main",
        "Big rect. left top",
        "Big rect. right bottom",
        "Big rect. right main",
        "Big rect. right top",
        "Circle central",
        "Circle left",
        "Circle right",
        "Goal left crossbar",
        "Goal left post left ",
        "Goal left post right",
        "Goal right crossbar",
        "Goal right post left",
        "Goal right post right",
        "Middle line",
        "Side line bottom",
        "Side line left",
        "Side line right",
        "Side line top",
        "Small rect. left bottom",
        "Small rect. left main",
        "Small rect. left top",
        "Small rect. right bottom",
        "Small rect. right main",
        "Small rect. right top"
    ]

    CALIBRATION_LABELS_BACKWARD = dict((v, k) for k, v in enumerate(CALIBRATION_LABELS))
    ALL_LABELS = range(len(CALIBRATION_LABELS))

    NOT_FIELD = 0
    BIG_RECT_LEFT_BOTTOM = 1
    BIG_RECT_LEFT_MAIN = 2
    BIG_RECT_LEFT_TOP = 3
    BIG_RECT_RIGHT_BOTTOM = 4
    BIG_RECT_RIGHT_MAIN = 5
    BIG_RECT_RIGHT_TOP = 6
    CIRCLE_CENTRAL = 7
    CIRCLE_LEFT = 8
    CIRCLE_RIGHT = 9
    GOAL_LEFT_CROSSBAR = 10
    GOAL_LEFT_POST_LEFT  = 11
    GOAL_LEFT_POST_RIGHT = 12
    GOAL_RIGHT_CROSSBAR = 13
    GOAL_RIGHT_POST_LEFT = 14
    GOAL_RIGHT_POST_RIGHT = 15
    MIDDLE_LINE = 16
    SIDE_LINE_BOTTOM = 17
    SIDE_LINE_LEFT = 18
    SIDE_LINE_RIGHT = 19
    SIDE_LINE_TOP = 20
    SMALL_RECT_LEFT_BOTTOM = 21
    SMALL_RECT_LEFT_MAIN = 22
    SMALL_RECT_LEFT_TOP = 23
    SMALL_RECT_RIGHT_BOTTOM = 24
    SMALL_RECT_RIGHT_MAIN = 25
    SMALL_RECT_RIGHT_TOP = 26

    @classmethod
    def forward(cls, idx: int):
        return cls.CALIBRATION_LABELS[idx]

    @classmethod
    def backward(cls, label: str):
        return cls.CALIBRATION_LABELS_BACKWARD.get(label, None)


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
        
    x_coord, y_coord, z_coord = 0, 1, 2
    if swap_coords:
        x_coord, y_coord = y_coord, x_coord

    after_mapping = np.array(after_mapping)
    after_mapping = torch.stack((
        torch.Tensor(after_mapping[:, x_coord]),
        torch.Tensor(after_mapping[:, y_coord]),
        torch.Tensor(after_mapping[:, z_coord])
    ), dim=1)

    # print('After mapping:')
    # print(after_mapping)
    # print(after_mapping.shape)
    return keypoint_idxs, after_mapping

def _homography_matrix(old_pts, new_pts):
    return HomographySolver()(old_pts, new_pts)