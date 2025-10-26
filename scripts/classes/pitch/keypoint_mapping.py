from typing import List

class PitchKeypointMapping:
    PITCH_LABELS: List[str] = [
        "centre",
        "circle_ten_yard_bottom",
        "circle_ten_yard_top",
        "circle_ten_yard_left",
        "circle_ten_yard_right",
        "field_top_left",
        "field_bottom_left",
        "field_top_right",
        "field_bottom_right",
        "midfield_top",
        "midfield_bottom",
        "left_penalty_top_left",
        "left_penalty_top_right",
        "left_penalty_bottom_left",
        "left_penalty_bottom_right",
        "right_penalty_top_left",
        "right_penalty_top_right",
        "right_penalty_bottom_left",
        "right_penalty_bottom_right",
        "left_goalkeeper_top_left",
        "left_goalkeeper_top_right",
        "left_goalkeeper_bottom_left",
        "left_goalkeeper_bottom_right",
        "right_goalkeeper_top_left",
        "right_goalkeeper_top_right",
        "right_goalkeeper_bottom_left",
        "right_goalkeeper_bottom_right",
    ]

    # 10-yard circle
    CENTRE = 0
    CIRCLE_TEN_YARD_BOTTOM = 1
    CIRCLE_TEN_YARD_TOP = 2
    CIRCLE_TEN_YARD_LEFT = 3
    CIRCLE_TEN_YARD_RIGHT = 4

    # Field
    FIELD_TOP_LEFT = 5
    FIELD_BOTTOM_LEFT = 6
    FIELD_TOP_RIGHT = 7
    FIELD_BOTTOM_RIGHT = 8
    MIDFIELD_TOP = 9
    MIDFIELD_BOTTOM = 10

    # 18-yard box
    LEFT_PENALTY_TOP_LEFT = 11
    LEFT_PENALTY_TOP_RIGHT = 12
    LEFT_PENALTY_BOTTOM_LEFT = 13
    LEFT_PENALTY_BOTTOM_RIGHT = 14
    RIGHT_PENALTY_TOP_LEFT = 15
    RIGHT_PENALTY_TOP_RIGHT = 16
    RIGHT_PENALTY_BOTTOM_LEFT = 17
    RIGHT_PENALTY_BOTTOM_RIGHT = 18

    # 6-yard box
    LEFT_GOALKEEPER_TOP_LEFT = 19
    LEFT_GOALKEEPER_TOP_RIGHT = 20
    LEFT_GOALKEEPER_BOTTOM_LEFT = 21
    LEFT_GOALKEEPER_BOTTOM_RIGHT = 22
    RIGHT_GOALKEEPER_TOP_LEFT = 23
    RIGHT_GOALKEEPER_TOP_RIGHT = 24
    RIGHT_GOALKEEPER_BOTTOM_LEFT = 25
    RIGHT_GOALKEEPER_BOTTOM_RIGHT = 26

    PITCH_LABELS_BACKWARD = dict((v, k) for k, v in enumerate(PITCH_LABELS))

    # Boxes
    BOX_FIELD = (FIELD_TOP_LEFT, FIELD_TOP_RIGHT, FIELD_BOTTOM_RIGHT, FIELD_BOTTOM_LEFT)
    BOX_FIELD_LEFT_HALF = (FIELD_TOP_LEFT, MIDFIELD_TOP, MIDFIELD_BOTTOM, FIELD_BOTTOM_LEFT)
    BOX_FIELD_RIGHT_HALF = (MIDFIELD_TOP, FIELD_TOP_RIGHT, FIELD_BOTTOM_RIGHT, MIDFIELD_BOTTOM)
    BOX_PENALTY_LEFT = (LEFT_PENALTY_TOP_LEFT, LEFT_PENALTY_TOP_RIGHT, LEFT_PENALTY_BOTTOM_RIGHT, LEFT_PENALTY_BOTTOM_LEFT)
    BOX_PENALTY_RIGHT = (RIGHT_PENALTY_TOP_LEFT, RIGHT_PENALTY_TOP_RIGHT, RIGHT_PENALTY_BOTTOM_RIGHT, RIGHT_PENALTY_BOTTOM_LEFT)
    BOX_GOALKEEPER_LEFT = (LEFT_GOALKEEPER_TOP_LEFT, LEFT_GOALKEEPER_TOP_RIGHT, LEFT_GOALKEEPER_BOTTOM_RIGHT, LEFT_GOALKEEPER_BOTTOM_LEFT)
    BOX_GOALKEEPER_RIGHT = (RIGHT_GOALKEEPER_TOP_LEFT, RIGHT_GOALKEEPER_TOP_RIGHT, RIGHT_GOALKEEPER_BOTTOM_RIGHT, RIGHT_GOALKEEPER_BOTTOM_LEFT)

    # Ellipses / Conics
    ELLIPSES_TEN_YARD = (CENTRE, CIRCLE_TEN_YARD_TOP, CIRCLE_TEN_YARD_BOTTOM, CIRCLE_TEN_YARD_LEFT, CIRCLE_TEN_YARD_RIGHT)

    @classmethod
    def forward(cls, idx: int):
        return cls.PITCH_LABELS[idx]

    @classmethod
    def backward(cls, label: str):
        return cls.PITCH_LABELS_BACKWARD.get(label, None)