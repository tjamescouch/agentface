"""Tests for the marker parser."""

from agentface.marker import parse, Marker


def test_basic_extraction():
    clean, markers, pending = parse("Hello <af:happy> world")
    assert clean == "Hello  world"
    assert markers == [Marker("happy", 1.0)]
    assert pending == ""


def test_intensity():
    clean, markers, _ = parse("Test <af:sad:0.5> text")
    assert markers == [Marker("sad", 0.5)]
    assert "<af:" not in clean


def test_multiple_markers():
    clean, markers, _ = parse("<af:happy:0.8>good<af:thinking>hmm")
    assert len(markers) == 2
    assert markers[0] == Marker("happy", 0.8)
    assert markers[1] == Marker("thinking", 1.0)
    assert clean == "goodhmm"


def test_unknown_expression_dropped():
    clean, markers, _ = parse("test <af:wiggle> text")
    assert markers == []
    assert clean == "test  text"


def test_intensity_clamped():
    _, markers, _ = parse("<af:happy:2.5>")
    assert markers[0].intensity == 1.0

    _, markers, _ = parse("<af:sad:-0.5>")
    # Negative doesn't match the regex pattern [0-9.], so it won't parse
    assert markers == []


def test_case_insensitive():
    _, markers, _ = parse("<AF:HAPPY:0.6>")
    assert markers == [Marker("happy", 0.6)]


def test_mid_word_extraction():
    clean, markers, _ = parse("hel<af:happy>lo")
    assert clean == "hello"
    assert markers == [Marker("happy", 1.0)]


def test_partial_tag_buffering():
    # First chunk ends with incomplete tag
    clean1, markers1, pending = parse("Hello <af:hap")
    assert markers1 == []
    assert pending == "<af:hap"
    assert clean1 == "Hello "

    # Second chunk completes the tag
    clean2, markers2, pending2 = parse("py:0.7> world", pending)
    assert markers2 == [Marker("happy", 0.7)]
    assert clean2 == " world"
    assert pending2 == ""


def test_clean_text_no_af_tags():
    text = "This <af:happy> is <af:sad:0.3> a <af:unknown> test"
    clean, _, _ = parse(text)
    assert "<af:" not in clean


def test_empty_input():
    clean, markers, pending = parse("")
    assert clean == ""
    assert markers == []
    assert pending == ""
