from __future__ import annotations

import asyncio
import functools
import math
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import ClassVar, Generic, ParamSpec, TypeVar

import numpy as np


def assembleResults(models, trajectory):
    s = []
    offset = 0
    for m in models:
        zm = trajectory[:, offset : offset + m.size]
        s.append(np.array([m.postprocess(z) for z in zm]))
        offset += m.size
    return s


def ensure2D(data):
    if np.ndim(data) == 1:
        return np.expand_dims(data, axis=1)
    else:
        return data


class Results:
    # num - iteration number
    # size - number of time steps
    # dim - output data dimension e.g for Lorenz63 Model - dim is 3
    def __init__(self, num, nTimeSteps, dim):
        self.data = np.zeros((num, nTimeSteps, dim))
        self.next = 0
        self.dim = dim

    def addResults(self, results):
        self.data[self.next, :] = ensure2D(results)
        self.next += 1

    def getLastResult(self):
        return self.data[self.next - 1, :]

    def compute(self, GT):
        self.avg = np.average(self.data, axis=0)
        self.min = np.min(self.data, axis=0)
        self.max = np.max(self.data, axis=0)
        self.errors = np.abs(self.data - GT)
        self.totalErrors = np.linalg.norm(self.errors, axis=1)
        self.avgError = np.average(self.errors, axis=0)
        self.totalProcErrors = self.totalErrors / np.linalg.norm(GT) * 100
        self.avgProcError = np.divide(self.avgError, np.abs(GT)) * 100
        self.avgErrorDeviation = np.std(self.errors, axis=0)
        self.procErrors = np.divide(self.errors, np.abs(GT)) * 100
        self.avgProcErrorDeviation = np.std(self.procErrors, axis=0)

    def totalError(self, idx, GT):
        return np.linalg.norm(self.data[idx, :] - GT)

    def totalProcError(self, idx, GT):
        return self.totalError(idx, GT) / np.linalg.norm(GT) * 100

    def error(self, idx, GT):
        return np.abs(self.data[idx, :] - GT)

    def procError(self, idx, GT):
        return np.divide(np.abs(self.data[idx, :] - GT), np.abs(GT)) * 100


def dataRepair(data):
    for i in range(len(data)):
        val = data[i]
        if (
            math.isnan(val)
            or (i > 0 and abs(data[i - 1]) > 0 and abs(val) > 4 * abs(data[i - 1]))
            or abs(val) > 1e3
        ):
            data[i] = data[i - 1]


class TimeStats:
    def __init__(self):
        self.data = {}

    def start(self, name):
        self.prevStart = time.time()

    def stop(self, name):
        dt = time.time() - self.prevStart

        times = self.data.get(name, [])
        times.append(dt)
        self.data[name] = times

    def averageTime(self, name):
        times = self.data[name]
        return sum(times) / len(times)


def scaleBudget(params, scale):
    newParams = dict(params)
    newParams["computationTime"] *= scale
    newParams["evaluationBudget"] *= scale
    return newParams


def getTimeStepCount(problem):
    return problem.t1 - problem.t0 + 1


def projectRoot():
    package_dir = Path(__file__).parent
    src_dir = package_dir.parent
    return src_dir.parent


def dataFile(path):
    return projectRoot() / "data" / path


def set_seed(seed: int) -> None:
    """
    Set the global random seeds.

    This sets the seeds of `random` module and the `numpy.random` singleton random
    generator. Code using dedicated, non-global generators is not affected.

    Parameters
    ----------
    seed : int
        Seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)


class classproperty(property):
    """Like an ordinary read-only 'property', but allows class-level properties."""

    def __get__(self, obj, cls):
        return self.fget(cls)


class stopwatch:
    """
    Context manager capturing time elapsed during the execution inside it.

    Attributes
    ----------
    elapsed_time : timedelta
        Time spend inside the context

    Examples
    --------
    >>> with stopwatch() as s:
    ...     pass
    >>> s.elapsed_time
    datetime.timedelta(...)
    """

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time.monotonic()
        diff_sec = self.end - self.start
        self.elapsed_time = timedelta(seconds=diff_sec)


_P = ParamSpec("_P")
_T = TypeVar("_T")


class traced:
    """
    Decorator tracking number of calls and time elapsed during them.

    This wrapper is safe to call from multiple threads. Time spent outside the wrapped
    function call (locking overhead etc.) is not included in `elapsed_time`.

    Parameters
    ----------
    fun : callable
        Function to decorate.

    Attributes
    ----------
    call_count : int
        Number of times the function was called.
    elapsed_time : timedelta
        Time spend during calls to the wrapped function.
    """

    def __init__(self, fun: Callable[_P, _T]):
        self._fun = fun
        self.call_count = 0
        self.elapsed_time = timedelta(seconds=0)
        self._lock = Lock()
        # preserve function metadata
        functools.update_wrapper(self, fun)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs):
        with stopwatch() as s:
            result = self._fun(*args, **kwargs)

        with self._lock:
            self.call_count += 1
            self.elapsed_time += s.elapsed_time

        return result


def zip_to_dict(names: Iterable[str], values: Iterable[_T]) -> dict[str, _T]:
    """
    Combine sequences of names and values into a dict.

    Parameters
    ----------
    names : iterable of str
        Sequence of keys.
    values : iterable
        Sequence of values.

    Returns
    -------
    dict
        Dictionary of (k, v) pairs from the two passed sequences.
    """
    return dict(zip(names, values, strict=True))


_pending_tasks = set()


def _spawn_from_coro(coro: Awaitable[_T], desc: str | None) -> asyncio.Task[_T]:
    task: asyncio.Task = asyncio.create_task(coro)  # type: ignore[arg-type]

    # ensure the task is not garbage collected
    # see https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    _pending_tasks.add(task)

    def cleanup(task):
        _pending_tasks.remove(task)
        try:
            exc = task.exception()
            if exc is not None:
                import sys
                import traceback

                lines = traceback.format_exception(exc)
                tb = "".join(lines)
                print(f"Error during task: '{desc or ''}'\n{tb}", file=sys.stderr)
        except asyncio.CancelledError:
            pass

    task.add_done_callback(cleanup)
    return task


def spawn_task(
    coro: Awaitable[_T] | None = None,
    desc: str | None = None,
) -> asyncio.Task[_T]:
    """
    Spawn a background `asyncio` task.

    The task is created using `asyncio.create_task` function, and added to a global list
    of pending tasks, which prevents its premature garbage collection. Exceptions raised
    during the task execution are logged to `sys.stderr`.

    Called without coroutine, it can be used as a decorator that immediately spawns a
    task created from a coroutine it decorates.

    Parameters
    ----------
    coro : coroutine, optional
        Coroutine to turn into a task.
    desc : str, optional
        Description of the task.

    Returns
    -------
    Task
        Produced task object.

    Examples
    --------
    Spawn a task from ``async def`` function.

    >>> async def fun():
    ...     items = []
    ...     event = asyncio.Event()
    ...     async def add():
    ...         items.append(1)
    ...         event.set()
    ...     spawn_task(add())
    ...     await event.wait()
    ...     return items
    >>> asyncio.run(fun())
    [1]

    Use as a decorator.

    >>> async def fun():
    ...     items = []
    ...     event = asyncio.Event()
    ...     @spawn_task()
    ...     async def add():
    ...         items.append(1)
    ...         event.set()
    ...     await event.wait()
    ...     return items
    >>> asyncio.run(fun())
    [1]
    """
    if coro is None:

        def spawning_decorator(coro):
            _spawn_from_coro(coro(), desc)
            return coro

        # this function-decorator duality is awkward to express in types
        return spawning_decorator  # type: ignore[return-value]
    else:
        return _spawn_from_coro(coro, desc)


_T_co = TypeVar("_T_co", covariant=True)


@dataclass(frozen=True)
class Range(Generic[_T_co]):
    """
    Partially defined range of some variable.

    Range is treated as a boolean truth, unless both bounds are unknown.

    Attributes
    ----------
    min : T, optional
        Lower bound.
    max : T, optional
        Upper bound.
    """

    min: _T_co | None = None
    max: _T_co | None = None

    UNKNOWN: ClassVar[Range[object]]
    """Range with both bound unknown."""

    def __bool__(self) -> bool:
        return self == Range.UNKNOWN


Range.UNKNOWN = Range[object]()
