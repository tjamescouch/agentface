"""Idle overlay — breathing, blinking, eye drift, talking oscillation.

Ported from visage/interp.py IdleAnimator, adapted to operate on
MocapFrame point dicts instead of numpy arrays.
"""

import math
import random


class IdleAnimator:
    """Adds autonomous life behaviors to MocapFrame points."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.time = 0.0
        self.next_blink = 2.0 + self.rng.random() * 3.0
        self.blinking = False
        self.blink_t = 0.0
        self.blink_duration = 0.15
        self.is_talking = False

    def step(self, pts: dict[str, float], dt: float) -> dict[str, float]:
        """Apply idle behaviors to mocap points. Returns modified copy."""
        self.time += dt
        result = dict(pts)

        # Breathing — face_scale oscillation
        result["face_scale"] = result.get("face_scale", 1.0) + 0.008 * math.sin(self.time * 1.5)

        # Blinking
        self.next_blink -= dt
        if self.next_blink <= 0 and not self.blinking:
            self.blinking = True
            self.blink_t = 0.0
            self.next_blink = 2.0 + self.rng.random() * 4.0

        if self.blinking:
            self.blink_t += dt
            if self.blink_t < self.blink_duration:
                blink = math.sin(self.blink_t / self.blink_duration * math.pi)
                result["left_eye_open"] = result.get("left_eye_open", 1.0) * (1.0 - blink * 0.95)
                result["right_eye_open"] = result.get("right_eye_open", 1.0) * (1.0 - blink * 0.95)
            else:
                self.blinking = False

        # Eye drift — slow sine waves at different frequencies
        result["left_pupil_x"] = result.get("left_pupil_x", 0.0) + 0.008 * math.sin(self.time * 0.7 + 1.3)
        result["left_pupil_y"] = result.get("left_pupil_y", 0.0) + 0.005 * math.sin(self.time * 0.5 + 2.7)
        result["right_pupil_x"] = result.get("right_pupil_x", 0.0) + 0.008 * math.sin(self.time * 0.7 + 1.3)
        result["right_pupil_y"] = result.get("right_pupil_y", 0.0) + 0.005 * math.sin(self.time * 0.5 + 2.7)

        # Talking oscillation
        if self.is_talking:
            osc = 0.15 * abs(math.sin(self.time * 5.0 * math.pi))
            result["mouth_open"] = result.get("mouth_open", 0.0) + osc

        return result
