from dataclasses import dataclass

import numpy as np
import pytest

from supy.problem import Event, Interval, Problem


def test_domain() -> None:
    problem = Problem(
        ground_truth={"val": np.array([1, 3, 7])},
        time_points=np.array([1, 2, 3]),
        learning_window=Interval(0.5, 2.3),
    )
    assert problem.domain == Interval(1, 3)


def test_time_and_ground_truth_must_match() -> None:
    with pytest.raises(ValueError, match="'val'") as excinfo:
        Problem(
            ground_truth={"val": np.array([1, 2, 3])},
            time_points=np.array([1, 2, 3, 4]),
            learning_window=Interval(0.5, 2.3),
        )

    msg = str(excinfo.value)
    assert "3 != 4" in msg


def test_ground_truth_must_be_1D() -> None:
    with pytest.raises(ValueError, match="'val'") as excinfo:
        Problem(
            ground_truth={"val": np.array([[1, 2], [3, 4], [5, 6]])},
            time_points=np.array([1, 2]),
            learning_window=Interval(0.5, 1.2),
        )

    msg = str(excinfo.value)
    assert "(3, 2)" in msg


def test_event_list_is_sorted() -> None:
    @dataclass(frozen=True)
    class TestEvent(Event):
        name: str

    event_a = TestEvent(0.2, "A")
    event_b = TestEvent(0.3, "B")
    event_c = TestEvent(1.2, "C")

    problem = Problem(
        ground_truth={"val": np.array([1, 3, 7])},
        time_points=np.array([1, 2, 3]),
        learning_window=Interval(0.5, 2.3),
        events=[event_c, event_a, event_b],
    )

    assert problem.events == [event_a, event_b, event_c]
