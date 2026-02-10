"""Tests for the signal merger."""

from agentface.merger import SignalMerger, EXPRESSION_NAMES, _IDX


def test_marker_sets_weight():
    m = SignalMerger()
    m.push_marker("happy", 0.8)
    vec = m.step(0.0)
    assert vec[_IDX["happy"]] >= 0.8


def test_marker_overrides_sentiment():
    m = SignalMerger()
    m.push_sentiment(-0.5, 0.0, False)  # sad sentiment
    m.push_marker("happy", 0.9)         # but happy marker
    vec = m.step(0.0)
    assert vec[_IDX["happy"]] >= vec[_IDX["sad"]]


def test_marker_decays():
    m = SignalMerger()
    m.push_marker("happy", 0.5)

    # Step forward in time
    for _ in range(20):
        vec = m.step(0.1)

    # After 2 seconds of decay at 0.3/s, should be near zero
    assert vec[_IDX["happy"]] < 0.1


def test_sentiment_positive_maps_happy():
    m = SignalMerger()
    m.push_sentiment(0.5, 0.2, False)
    vec = m.step(0.01)
    assert vec[_IDX["happy"]] > 0


def test_sentiment_negative_maps_sad():
    m = SignalMerger()
    m.push_sentiment(-0.5, 0.2, False)
    vec = m.step(0.01)
    assert vec[_IDX["sad"]] > 0


def test_talking_flag():
    m = SignalMerger()
    m.push_sentiment(0.0, 0.1, True)
    vec = m.step(0.01)
    assert vec[_IDX["talking"]] > 0


def test_vector_normalized():
    m = SignalMerger()
    m.push_marker("happy", 1.0)
    m.push_marker("sad", 1.0)
    m.push_marker("surprised", 1.0)
    vec = m.step(0.0)
    assert sum(vec) <= 1.01  # allow tiny float error
