"""Signal merger — blends marker signals and sentiment into an expression vector."""

from dataclasses import dataclass, field

EXPRESSION_NAMES = [
    "happy", "sad", "thinking", "surprised",
    "confused", "angry", "neutral", "talking",
]
EXPRESSION_DIM = len(EXPRESSION_NAMES)
_IDX = {name: i for i, name in enumerate(EXPRESSION_NAMES)}


@dataclass
class _MarkerSignal:
    expression: str
    intensity: float
    age: float = 0.0
    decay_rate: float = 0.3  # weight units per second


class SignalMerger:
    """Combines marker and sentiment signals into a single expression vector."""

    def __init__(self):
        self._markers: list[_MarkerSignal] = []
        self._sentiment_weights: list[float] = [0.0] * EXPRESSION_DIM
        self._arousal: float = 0.0
        self._talking: bool = False

    def push_marker(self, expression: str, intensity: float):
        """Inject a marker-driven expression signal."""
        if expression not in _IDX:
            return
        # Replace existing marker for same expression
        self._markers = [m for m in self._markers if m.expression != expression]
        self._markers.append(_MarkerSignal(
            expression=expression,
            intensity=min(1.0, max(0.0, intensity)),
        ))

    def push_sentiment(self, valence: float, arousal: float, talking: bool):
        """Update the ambient sentiment signal."""
        self._arousal = arousal
        self._talking = talking

        w = [0.0] * EXPRESSION_DIM
        # Map valence/arousal to expression weights
        if valence > 0.05:
            w[_IDX["happy"]] = valence
        elif valence < -0.05:
            w[_IDX["sad"]] = abs(valence)
            if valence < -0.3:
                w[_IDX["angry"]] = (abs(valence) - 0.3) * 0.5

        if arousal > 0.3:
            w[_IDX["surprised"]] = (arousal - 0.3) * 0.5

        # Thinking: low arousal, near-neutral valence
        if arousal > 0.1 and abs(valence) < 0.2:
            w[_IDX["thinking"]] = arousal * 0.5

        if talking:
            w[_IDX["talking"]] = 0.3

        self._sentiment_weights = w

    def step(self, dt: float) -> list[float]:
        """Advance decay and return blended expression vector."""
        # Decay markers
        for m in self._markers:
            m.age += dt
            m.intensity = max(0.0, m.intensity - m.decay_rate * dt)
        self._markers = [m for m in self._markers if m.intensity > 0.001]

        # Start with sentiment as base
        result = list(self._sentiment_weights)

        # Markers override sentiment for their dimension
        for m in self._markers:
            idx = _IDX[m.expression]
            result[idx] = max(result[idx], m.intensity)

        # Normalize so weights sum to 1 (or less if all near zero)
        total = sum(result)
        if total > 1.0:
            result = [w / total for w in result]

        return result
