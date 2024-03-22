import asyncio
from datetime import timedelta

import pytest

from supy.utils import spawn_task, stopwatch, traced, zip_to_dict


def test_traced_calls_wrapped_function_correctly():
    @traced
    def f(x, y, z):
        return x + y * z

    assert f(1, 2, 3) == 7
    assert f(1, z=3, y=2) == 7


def test_traced_respects_param_passing_spec():
    @traced
    def f(a, b, /, c, *, d, e):
        return True

    assert f(1, 2, 3, d=4, e=5)
    assert f(1, 2, c=3, d=4, e=5)

    with pytest.raises(TypeError, match=r"positional-only .* keyword"):
        assert f(1, b=2, c=3, d=4, e=5)

    with pytest.raises(TypeError, match=r"3 positional .* 4 positional"):
        assert f(1, 2, 3, 4, e=5)


def test_traced_reports_call_count():
    @traced
    def f(x, y):
        return x + y

    f(1, 2)
    f(3, 4)

    assert f.call_count == 2


class TimeMachine:
    def __init__(self, time=0):
        self.time = time

    def tell(self):
        return self.time

    def advance(self, delta_sec):
        self.time += delta_sec


@pytest.fixture()
def custom_time(mocker):
    time = TimeMachine()
    mocker.patch("time.monotonic", time.tell)
    return time


def test_stopwatch(custom_time):
    with stopwatch() as s:
        custom_time.advance(5)

    assert s.elapsed_time == timedelta(seconds=5)


def test_traced_reports_time_elapsed(custom_time):
    @traced
    def f(delay):
        custom_time.advance(delay)

    f(2)
    f(7)

    assert f.elapsed_time == timedelta(seconds=9)


def test_traced_preserves_function_info():
    @traced
    def f(a: int, b: list[int]):
        """Test function"""

    assert f.__name__ == "f"
    assert f.__doc__ == "Test function"
    assert f.__annotations__ == {"a": int, "b": list[int]}


def test_zip_to_dict_works_for_equal_lengths():
    result = zip_to_dict(["a", "b", "c"], [1, 2, 3])

    assert result == dict(a=1, b=2, c=3)


def test_zip_to_dict_fails_for_different_lengths():
    with pytest.raises(ValueError, match="argument 2 is longer than argument 1"):
        zip_to_dict(["a", "b", "c"], [1, 2, 3, 4])


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_spawn_task():
    items = []
    event = asyncio.Event()

    async def append():
        items.append(1)
        items.append(2)
        event.set()

    spawn_task(append())

    await event.wait()
    assert items == [1, 2]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_spawn_task_decorator():
    items = []
    event = asyncio.Event()

    @spawn_task()
    async def append():
        items.append(1)
        items.append(2)
        event.set()

    await event.wait()
    assert items == [1, 2]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_spawn_task_decorator_with_desc():
    items = []
    event = asyncio.Event()

    @spawn_task(desc="some task")
    async def append():
        items.append(1)
        items.append(2)
        event.set()

    await event.wait()
    assert items == [1, 2]
