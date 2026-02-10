"""Tests for the sentiment brain."""

from agentface.sentiment import SentimentBrain


def test_positive_text_increases_valence():
    brain = SentimentBrain()
    brain.feed("This is great wonderful excellent", 1.0)
    assert brain.emotion.valence > 0.1


def test_negative_text_decreases_valence():
    brain = SentimentBrain()
    brain.feed("error crash fail broken terrible", 1.0)
    assert brain.emotion.valence < -0.1


def test_neutral_text_stays_near_zero():
    brain = SentimentBrain()
    brain.feed("the cat sat on the mat", 1.0)
    assert abs(brain.emotion.valence) < 0.1


def test_decay_in_silence():
    brain = SentimentBrain()
    brain.feed("great wonderful awesome", 1.0)
    initial_v = brain.emotion.valence
    assert initial_v > 0

    # Simulate 5 seconds of silence
    for _ in range(50):
        brain.step(0.1, 6.0)

    assert abs(brain.emotion.valence) < abs(initial_v) * 0.5


def test_talking_detection():
    brain = SentimentBrain()
    brain.feed("hello", 1.0)
    brain.step(0.01, 1.01)
    assert brain.emotion.talking is True

    # After silence
    brain.step(0.01, 2.0)
    assert brain.emotion.talking is False


def test_valence_clamped():
    brain = SentimentBrain()
    for _ in range(20):
        brain.feed("great excellent perfect wonderful", 1.0)
    assert brain.emotion.valence <= 1.0
    assert brain.emotion.valence >= -1.0


def test_arousal_clamped():
    brain = SentimentBrain()
    for _ in range(20):
        brain.feed("! wow crash danger exciting", 1.0)
    assert brain.emotion.arousal <= 1.0
    assert brain.emotion.arousal >= 0.0
