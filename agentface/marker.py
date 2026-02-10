"""Marker parser — extracts <af:expression:intensity> tags from token stream."""

import re
from dataclasses import dataclass

KNOWN_EXPRESSIONS = frozenset({
    "happy", "sad", "thinking", "surprised",
    "confused", "angry", "neutral", "talking",
})

_TAG_RE = re.compile(r"<af:(\w+)(?::([0-9]*\.?[0-9]+))?>", re.IGNORECASE)

# Detects an incomplete tag at the end of a chunk (potential partial)
_PARTIAL_RE = re.compile(r"<af[^>]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Marker:
    expression: str
    intensity: float


def parse(text: str, pending: str = "") -> tuple[str, list[Marker], str]:
    """Parse text for af markers.

    Args:
        text: Raw text chunk (may contain partial tags).
        pending: Leftover from a previous partial tag.

    Returns:
        (clean_text, markers, new_pending)
        - clean_text: text with all af tags stripped.
        - markers: list of extracted Marker objects.
        - new_pending: any incomplete tag at the end to carry forward.
    """
    combined = pending + text
    markers: list[Marker] = []

    # Extract complete tags
    def _replace(m: re.Match) -> str:
        name = m.group(1).lower()
        raw_intensity = m.group(2)
        if name not in KNOWN_EXPRESSIONS:
            return ""  # silently drop unknown
        intensity = min(1.0, max(0.0, float(raw_intensity))) if raw_intensity else 1.0
        markers.append(Marker(expression=name, intensity=intensity))
        return ""

    clean = _TAG_RE.sub(_replace, combined)

    # Check for incomplete tag at the end
    new_pending = ""
    partial = _PARTIAL_RE.search(clean)
    if partial:
        new_pending = clean[partial.start():]
        clean = clean[:partial.start()]

    return clean, markers, new_pending
