"""Frame emitter — packages and delivers MocapFrame at steady FPS."""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from agentface.expression_net import MOCAP_POINTS, NEUTRAL


@dataclass
class MocapFrame:
    t: float
    pts: dict[str, float]

    def to_json(self) -> str:
        return json.dumps({"t": round(self.t, 4), "pts": self.pts}, separators=(",", ":"))


class FrameEmitter:
    """Emits MocapFrame at a steady rate."""

    def __init__(self, fps: int = 30, callback: Optional[Callable[[MocapFrame], None]] = None):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self._accumulator = 0.0
        self._callback = callback
        self._start_time = time.monotonic()

    def emit(self, pts: dict[str, float], timestamp: float) -> Optional[MocapFrame]:
        """Package points into a MocapFrame and deliver it.

        Returns the frame if emitted, None if skipped (not time yet).
        """
        frame = MocapFrame(t=timestamp, pts={k: round(pts.get(k, NEUTRAL.get(k, 0.0)), 6) for k in MOCAP_POINTS})

        if self._callback:
            self._callback(frame)
        else:
            sys.stdout.write(frame.to_json() + "\n")
            sys.stdout.flush()

        return frame

    def should_emit(self, dt: float) -> bool:
        """Check if enough time has passed to emit a frame."""
        self._accumulator += dt
        if self._accumulator >= self.frame_interval:
            self._accumulator -= self.frame_interval
            return True
        return False
