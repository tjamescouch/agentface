"""Expression net — hand-crafted mapping from expression vector to MocapFrame points.

This is the fallback mapping used before a trained MLP is available.
Expression templates are ported from visage/face.py EXPRESSIONS dict,
translated to MocapFrame control point names.
"""

from agentface.merger import EXPRESSION_NAMES

# 18 MocapFrame control points with neutral defaults
MOCAP_POINTS = [
    "left_eye_open", "right_eye_open",
    "left_pupil_x", "left_pupil_y",
    "right_pupil_x", "right_pupil_y",
    "left_brow_height", "left_brow_angle",
    "right_brow_height", "right_brow_angle",
    "mouth_open", "mouth_wide", "mouth_smile",
    "jaw_open", "face_scale",
    "head_pitch", "head_yaw", "head_roll",
]

NEUTRAL = {
    "left_eye_open": 1.0, "right_eye_open": 1.0,
    "left_pupil_x": 0.0, "left_pupil_y": 0.0,
    "right_pupil_x": 0.0, "right_pupil_y": 0.0,
    "left_brow_height": 0.0, "left_brow_angle": 0.0,
    "right_brow_height": 0.0, "right_brow_angle": 0.0,
    "mouth_open": 0.0, "mouth_wide": 0.0, "mouth_smile": 0.0,
    "jaw_open": 0.0, "face_scale": 1.0,
    "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
}

# Expression templates — deltas from NEUTRAL for each named expression
# Ported from visage/face.py EXPRESSIONS, mapped to MocapFrame point names
_TEMPLATES: dict[str, dict[str, float]] = {
    "happy": {
        "left_eye_open": 0.85, "right_eye_open": 0.85,
        "mouth_wide": 0.04, "mouth_smile": 0.25, "mouth_open": 0.08,
    },
    "sad": {
        "left_eye_open": 0.6, "right_eye_open": 0.6,
        "mouth_smile": -0.25,
        "left_brow_angle": -0.12, "left_brow_height": -0.02,
        "right_brow_angle": 0.12, "right_brow_height": -0.02,
    },
    "thinking": {
        "left_eye_open": 0.7, "right_eye_open": 0.7,
        "left_pupil_x": 0.02, "left_pupil_y": -0.01,
        "right_pupil_x": 0.02, "right_pupil_y": -0.01,
        "mouth_smile": -0.08,
        "left_brow_angle": 0.12, "left_brow_height": 0.03,
        "right_brow_angle": -0.04, "right_brow_height": 0.01,
    },
    "surprised": {
        "left_eye_open": 1.3, "right_eye_open": 1.3,
        "mouth_open": 0.45, "mouth_wide": -0.03,
        "left_brow_height": 0.06, "right_brow_height": 0.06,
    },
    "confused": {
        "left_eye_open": 0.8, "right_eye_open": 0.6,
        "left_pupil_x": -0.01, "right_pupil_x": 0.01,
        "mouth_smile": -0.12, "mouth_open": 0.04,
        "left_brow_angle": 0.18, "left_brow_height": 0.04,
        "right_brow_angle": -0.12, "right_brow_height": -0.01,
    },
    "angry": {
        "left_eye_open": 0.7, "right_eye_open": 0.7,
        "mouth_smile": -0.2, "mouth_open": 0.05,
        "left_brow_angle": 0.15, "left_brow_height": -0.03,
        "right_brow_angle": -0.15, "right_brow_height": -0.03,
    },
    "neutral": {},
    "talking": {
        "mouth_open": 0.35, "mouth_smile": 0.03,
    },
}


def forward(expression_vector: list[float]) -> dict[str, float]:
    """Map expression vector to MocapFrame control points.

    Args:
        expression_vector: 8 floats, one weight per expression class
            (same order as EXPRESSION_NAMES).

    Returns:
        Dict of 18 MocapFrame point names → values.
    """
    pts = dict(NEUTRAL)

    for i, weight in enumerate(expression_vector):
        if weight < 0.001:
            continue
        name = EXPRESSION_NAMES[i]
        template = _TEMPLATES.get(name, {})
        for pt, val in template.items():
            neutral_val = NEUTRAL[pt]
            # Blend toward template value weighted by expression weight
            pts[pt] += (val - neutral_val) * weight

    return pts
