"""Tests for the full pipeline."""

import json

from agentface.pipeline import AgentFacePipeline
from agentface.expression_net import MOCAP_POINTS


def test_end_to_end():
    frames = []
    pipeline = AgentFacePipeline(fps=30, callback=lambda f: frames.append(f))

    # Feed text with markers
    clean = pipeline.feed("Hello! This is great <af:happy:0.8> wonderful news", 1.0)

    # Markers stripped
    assert "<af:" not in clean
    assert "wonderful" in clean

    # Step to emit a frame
    frame = pipeline.step(0.034, 1.034)
    assert frame is not None

    # All 18 points present
    assert set(frame.pts.keys()) == set(MOCAP_POINTS)


def test_json_output():
    frames = []
    pipeline = AgentFacePipeline(fps=30, callback=lambda f: frames.append(f))
    pipeline.feed("great", 1.0)
    pipeline.step(0.034, 1.034)

    assert len(frames) == 1
    j = json.loads(frames[0].to_json())
    assert "t" in j
    assert "pts" in j
    assert len(j["pts"]) == 18


def test_happy_text_produces_smile():
    frames = []
    pipeline = AgentFacePipeline(fps=30, callback=lambda f: frames.append(f))
    pipeline.feed("This is wonderful great excellent happy", 1.0)
    pipeline.step(0.034, 1.034)

    assert len(frames) == 1
    assert frames[0].pts["mouth_smile"] > 0


def test_silence_decays_to_neutral():
    frames = []
    pipeline = AgentFacePipeline(fps=30, callback=lambda f: frames.append(f))
    pipeline.feed("great wonderful", 1.0)
    pipeline.step(0.034, 1.034)

    initial_smile = frames[-1].pts["mouth_smile"]

    # Simulate 5 seconds of silence
    for i in range(150):
        t = 1.034 + (i + 1) * 0.034
        pipeline.step(0.034, t)

    final_smile = frames[-1].pts["mouth_smile"]
    assert abs(final_smile) < abs(initial_smile)


def test_idle_behaviors_active():
    frames = []
    pipeline = AgentFacePipeline(fps=30, callback=lambda f: frames.append(f))

    # Just step without feeding text — idle should still modulate
    scales = []
    for i in range(60):
        pipeline.step(0.034, i * 0.034)
        if frames:
            scales.append(frames[-1].pts["face_scale"])

    # Breathing should cause face_scale variation
    assert len(set(round(s, 4) for s in scales)) > 1
