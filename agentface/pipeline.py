"""Pipeline — wires marker parser, sentiment brain, merger, expression net, idle, and emitter."""

import time
from typing import Callable, Optional

from agentface import marker
from agentface.sentiment import SentimentBrain
from agentface.merger import SignalMerger
from agentface import expression_net
from agentface.idle import IdleAnimator
from agentface.emitter import FrameEmitter, MocapFrame


class AgentFacePipeline:
    """Full agentface pipeline: text in → MocapFrame out."""

    def __init__(
        self,
        fps: int = 30,
        callback: Optional[Callable[[MocapFrame], None]] = None,
    ):
        self.brain = SentimentBrain()
        self.merger = SignalMerger()
        self.idle = IdleAnimator()
        self.emitter = FrameEmitter(fps=fps, callback=callback)
        self._pending = ""  # partial marker tag buffer
        self._last_clean_text = ""

    def feed(self, text: str, timestamp: Optional[float] = None):
        """Feed raw text (may contain <af:...> markers).

        Returns clean text with markers stripped.
        """
        ts = timestamp or time.time()

        # Parse markers
        clean, markers, self._pending = marker.parse(text, self._pending)
        self._last_clean_text = clean

        # Feed clean text to sentiment brain
        if clean.strip():
            self.brain.feed(clean, ts)

        # Push markers to merger
        for m in markers:
            self.merger.push_marker(m.expression, m.intensity)

        return clean

    def step(self, dt: float, timestamp: Optional[float] = None) -> Optional[MocapFrame]:
        """Advance one time step. Emits a frame if the FPS timer is ready.

        Args:
            dt: Time delta since last step (seconds).
            timestamp: Current time (defaults to time.time()).

        Returns:
            MocapFrame if emitted, None if skipped.
        """
        ts = timestamp or time.time()

        # Advance sentiment
        self.brain.step(dt, ts)

        # Push sentiment to merger
        e = self.brain.emotion
        self.merger.push_sentiment(e.valence, e.arousal, e.talking)

        # Get blended expression vector
        expr_vec = self.merger.step(dt)

        # Map to mocap points
        pts = expression_net.forward(expr_vec)

        # Apply idle overlay
        self.idle.is_talking = e.talking
        pts = self.idle.step(pts, dt)

        # Emit frame if ready
        if self.emitter.should_emit(dt):
            return self.emitter.emit(pts, ts)

        return None
