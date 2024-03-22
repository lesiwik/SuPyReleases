from supy.interval import Interval


def test_length_computed_correctly():
    span = Interval(-2.0, 1.0)
    assert span.length == 3


def test_string_representation():
    span = Interval(2, 5.5)
    assert str(span) == "[2, 5.5]"


def test_can_deconstruct():
    span = Interval(2, 5.5)
    a, b = span
    assert a == 2
    assert b == 5.5


def test_can_check_containment():
    span = Interval(2.1, 3.5)
    assert 2.5 in span
    assert 4 not in span


def test_interval_is_closed():
    span = Interval(2.1, 3.5)
    assert 2.1 in span
    assert 3.5 in span
