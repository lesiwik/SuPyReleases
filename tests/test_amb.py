from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pytest

from supy.amb import (
    Aborted,
    Choice,
    Done,
    EmptyChoice,
    Event,
    ExecutionEnvironment,
    Failed,
    Finished,
    NodeClosed,
    NodeOpened,
    Path,
    Thunk,
    nondeterministic,
    once,
)

# ruff: noqa: F841


def local_vars(stacklevel=1):
    assert stacklevel > 0, "stack level must be positive"

    import inspect

    base_frame = inspect.currentframe()
    try:
        frame = base_frame
        for _ in range(stacklevel):
            frame = frame.f_back

        return frame.f_locals
    finally:
        del base_frame


class CallParamTracker:
    def __init__(self):
        self.log = []

    def record(self, env=None, **kwargs):
        args = env or kwargs or local_vars(stacklevel=2)
        relevant = self._filter(args)
        self.log.append(relevant)

    def _filter(self, args):
        # remove self and context objects
        def should_skip(name, val):
            return val is self or name == "amb"

        return {k: v for k, v in args.items() if not should_skip(k, v)}

    def __getitem__(self, name):
        return [call[name] for call in self.log if name in call]

    @property
    def count(self):
        return len(self.log)


@pytest.fixture()
def calls():
    return CallParamTracker()


@pytest.fixture()
def some_exception(mocker):
    """
    Factory of objects equal to exceptions of given type and containing given message.

    >>> some_exception = getfixture('some_exception')
    >>> spec = some_exception(ValueError, "foo")
    >>> ValueError("foo is broken") == spec
    True
    """

    def factory(class_, message):
        def new_eq(self, e):
            return isinstance(e, class_) and message in str(e)

        m = mocker.MagicMock()
        m.__eq__ = new_eq

        m.__repr__ = lambda self: f"<{class_.__name__} with '{message}'>"
        return m

    return factory


def run(generator):
    for _ in generator:
        pass


def test_executes_function(calls):
    def fun(amb):
        i = amb("i", [1])
        calls.record()

    run(nondeterministic(fun))

    assert calls.count == 1


def test_executes_function_without_choice(calls):
    def fun(amb):
        calls.record()

    run(nondeterministic(fun))

    assert calls.count == 1


def test_aborts_function_with_empty_choice(calls):
    def fun(amb):
        i = amb("i", [])
        calls.record()

    run(nondeterministic(fun))

    assert calls.count == 0


def test_supplies_correct_value(calls):
    def fun(amb):
        i = amb("i", [5])
        calls.record()

    run(nondeterministic(fun))

    assert calls["i"] == [5]


def test_multiple_values_single_var(calls):
    def fun(amb):
        i = amb("i", [5, 2, 1])
        calls.record()

    run(nondeterministic(fun))

    assert calls["i"] == [5, 2, 1]


def test_single_value_multiple_vars(calls):
    def fun(amb):
        a = amb("a", [1])
        b = amb("b", [2])
        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [{"a": 1, "b": 2}]


def test_multiple_values_multiple_vars(calls):
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", ["foo", "bar"])
        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": 1, "b": "foo"},
        {"a": 1, "b": "bar"},
        {"a": 2, "b": "foo"},
        {"a": 2, "b": "bar"},
    ]


def test_can_abort(calls):
    def fun(amb):
        a = amb("a", [1, 2, 3, 4])
        if a % 2 == 1:
            amb.abort()
        calls.record()

    run(nondeterministic(fun))

    assert calls["a"] == [2, 4]


def test_different_values_in_branches(calls):
    def fun(amb):
        a = amb("a", [1, 2])

        if a == 1:
            b = amb("b", ["foo", "bar"])
        if a == 2:
            b = amb("b", ["xxx", "yyy"])

        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": 1, "b": "foo"},
        {"a": 1, "b": "bar"},
        {"a": 2, "b": "xxx"},
        {"a": 2, "b": "yyy"},
    ]


def test_different_branches(calls):
    def fun(amb):
        a = amb("a", ["foo", "bar", "zyx"])
        match a:
            case "foo":
                b = amb("b", [1, 2])
                c = amb("c", ["x", "y"])
            case "bar":
                d = amb("d", [3, 4])

        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": "foo", "b": 1, "c": "x"},
        {"a": "foo", "b": 1, "c": "y"},
        {"a": "foo", "b": 2, "c": "x"},
        {"a": "foo", "b": 2, "c": "y"},
        {"a": "bar", "d": 3},
        {"a": "bar", "d": 4},
        {"a": "zyx"},
    ]


def test_dependence_between_variables(calls):
    def fun(amb):
        a = amb("a", range(1, 5))
        b = amb("b", range(1, a))
        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": 2, "b": 1},
        {"a": 3, "b": 1},
        {"a": 3, "b": 2},
        {"a": 4, "b": 1},
        {"a": 4, "b": 2},
        {"a": 4, "b": 3},
    ]


def test_aborts_when_ensure_fails(calls):
    def fun(amb):
        a = amb("a", range(1, 6))
        b = amb("b", range(1, 6))
        c = amb("c", range(1, 6))
        amb.ensure(a**2 + b**2 == c**2)
        calls.record()

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": 3, "b": 4, "c": 5},
        {"a": 4, "b": 3, "c": 5},
    ]


def test_can_propagate_exceptions(calls):
    def fun(amb):
        a = amb("a", [1, 2, 3, 4])
        if a == 3:
            raise ValueError("this is bad")

        calls.record()

    with pytest.raises(ValueError, match="this is bad"):
        run(nondeterministic(fun, pass_exceptions=True))

    assert calls["a"] == [1, 2]


def test_thunks_share_values(calls, mocker):
    mock = mocker.MagicMock()

    def fun(amb):
        a = amb("a", [once(mock)])
        b = amb("b", [1, 2, 3])
        calls.record()

    run(nondeterministic(fun))

    assert calls.count == 3
    mock.assert_called_once()


def test_can_have_many_thunk_options(mocker):
    mock_1 = mocker.MagicMock()
    mock_2 = mocker.MagicMock()

    def fun(amb):
        a = amb("a", [once(mock_1), once(mock_2)])
        b = amb("b", [1, 2, 3])

    run(nondeterministic(fun))

    mock_1.assert_called_once()
    mock_2.assert_called_once()


def test_can_mix_thunk_and_regular(calls, mocker):
    mock = mocker.MagicMock(return_value=5)

    def fun(amb):
        a = amb("a", [once(mock), 3])
        b = amb("b", [1, 2])
        calls.record(a=a, b=b)

    run(nondeterministic(fun))

    mock.assert_called_once()
    assert calls.log == [
        {"a": 5, "b": 1},
        {"a": 5, "b": 2},
        {"a": 3, "b": 1},
        {"a": 3, "b": 2},
    ]


def test_can_have_multiple_thunk_variables(calls, mocker):
    foo = mocker.MagicMock(return_value="foo")
    bar = mocker.MagicMock(return_value="bar")

    def fun(amb):
        a = amb("a", [once(foo)])
        b = amb("b", [once(bar)])
        c = amb("c", [1, 2, 3])
        calls.record(a=a, b=b, c=c)

    run(nondeterministic(fun))

    foo.assert_called_once()
    bar.assert_called_once()
    assert calls.log == [
        {"a": "foo", "b": "bar", "c": 1},
        {"a": "foo", "b": "bar", "c": 2},
        {"a": "foo", "b": "bar", "c": 3},
    ]


def test_can_pass_args_to_thunk(mocker):
    mock = mocker.MagicMock()

    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [once(mock, a)])
        calls.record()

    run(nondeterministic(fun))

    mock.assert_has_calls([mocker.call(1), mocker.call(2)])


def test_handles_thunk_exceptiions(calls, mocker):
    bomb = mocker.MagicMock(side_effect=ValueError("boom"))

    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [once(bomb), 3])
        calls.record(a=a, b=b)

    run(nondeterministic(fun))

    assert calls.log == [
        {"a": 1, "b": 3},
        {"a": 2, "b": 3},
    ]


def test_sicp_amb_example(calls):
    # SICP 4.3.2 Multiple Dwellings
    floors = (0, 1, 2, 3, 4)

    def fun(amb):
        b = amb("Baker", floors)
        c = amb("Cooper", floors)
        f = amb("Fletcher", floors)
        m = amb("Miller", floors)
        s = amb("Smith", floors)

        amb.ensure(len({b, c, f, m, s}) == 5)  # different floors
        amb.ensure(b != 4)  # Baker is not on top floor
        amb.ensure(c != 0)  # Cooper is not on bottom floor
        amb.ensure(f not in [0, 4])  # Fletcher is not on top or bottom floor
        amb.ensure(m > c)  # Miller is above Cooper
        amb.ensure(s not in [f - 1, f + 1])  # Smith is not adjacent to Fletcher
        amb.ensure(f not in [c - 1, c + 1])  # Fletcher is not adjacent to Cooper

        calls.record(b=b, c=c, f=f, m=m, s=s)

    run(nondeterministic(fun))

    assert calls.log == [{"b": 2, "c": 1, "f": 3, "m": 4, "s": 0}]


def test_reports_results():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [5, 6])
        return a + b

    events = set(nondeterministic(fun))

    results = {
        Done(path=Path([("a", 0, 1), ("b", 0, 5)]), result=6),
        Done(path=Path([("a", 0, 1), ("b", 1, 6)]), result=7),
        Done(path=Path([("a", 1, 2), ("b", 0, 5)]), result=7),
        Done(path=Path([("a", 1, 2), ("b", 1, 6)]), result=8),
    }

    assert results <= events


def test_reports_exceptions():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [10, 20])
        if a + b == 21:
            raise ValueError("Oh no")

    events = set(nondeterministic(fun))
    by_path = {event.path: event for event in events}

    failed_path = Path([("a", 0, 1), ("b", 1, 20)])
    event = by_path[failed_path]

    assert isinstance(event, Failed)
    with pytest.raises(ValueError, match="Oh no"):
        raise event.exc


def test_does_not_report_aborted_by_default():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [1, 3])
        amb.ensure(a < b)

    events = set(nondeterministic(fun))

    aborted = Aborted(Path([("a", 1, 2), ("b", 0, 1)]))
    assert aborted not in events


def test_can_report_aborted():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [1, 3])
        amb.ensure(a < b)

    events = set(nondeterministic(fun, aborted=True))

    aborted = Aborted(Path([("a", 1, 2), ("b", 0, 1)]))
    assert aborted in events


def test_can_report_aborted_with_reason():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [1, 3])
        amb.ensure(a < b, "too large")

    events = set(nondeterministic(fun, aborted=True))

    aborted = Aborted(Path([("a", 1, 2), ("b", 0, 1)]), reason="too large")
    assert aborted in events


def test_does_not_report_node_events_by_default():
    def fun(amb):
        a = amb("a", [1])
        return a

    events = list(nondeterministic(fun))

    assert events == [Done(Path([("a", 0, 1)]), result=1)]


def test_reports_node_events_without_choice():
    def fun(amb):
        return 2

    events = list(nondeterministic(fun, node_events=True))

    assert events == [
        NodeOpened(Path()),
        Done(Path(), result=2),
        NodeClosed(Path()),
    ]


def test_reports_node_events_with_empty_choice():
    def fun(amb):
        amb("i", [])

    events = list(nondeterministic(fun, node_events=True, aborted=True))

    assert events == [
        NodeOpened(Path()),
        EmptyChoice(Path(), reason="empty choice"),
        NodeClosed(Path()),
    ]


def test_reports_node_events_single_var():
    def fun(amb):
        a = amb("a", [1, 2])
        return a

    events = list(nondeterministic(fun, node_events=True))

    assert events == [
        NodeOpened(Path()),
        #
        NodeOpened(Path([("a", 0, 1)])),
        Done(Path([("a", 0, 1)]), result=1),
        NodeClosed(Path([("a", 0, 1)])),
        #
        NodeOpened(Path([("a", 1, 2)])),
        Done(Path([("a", 1, 2)]), result=2),
        NodeClosed(Path([("a", 1, 2)])),
        #
        NodeClosed(Path()),
    ]


def test_reports_node_events_with_abort():
    def fun(amb):
        a = amb("a", [1, 2, 3])
        if a == 2:
            amb.abort()
        return a

    events = list(nondeterministic(fun, node_events=True, aborted=True))

    assert events == [
        NodeOpened(Path()),
        #
        NodeOpened(Path([("a", 0, 1)])),
        Done(Path([("a", 0, 1)]), result=1),
        NodeClosed(Path([("a", 0, 1)])),
        #
        NodeOpened(Path([("a", 1, 2)])),
        Aborted(Path([("a", 1, 2)])),
        NodeClosed(Path([("a", 1, 2)])),
        #
        NodeOpened(Path([("a", 2, 3)])),
        Done(Path([("a", 2, 3)]), result=3),
        NodeClosed(Path([("a", 2, 3)])),
        #
        NodeClosed(Path()),
    ]


def test_reports_node_events_dependence_between_variables():
    def fun(amb):
        a = amb("a", range(1, 4))
        b = amb("b", range(1, a))
        return a + b

    events = list(nondeterministic(fun, node_events=True, aborted=True))

    assert events == [
        NodeOpened(Path()),
        #
        NodeOpened(Path([("a", 0, 1)])),
        EmptyChoice(Path([("a", 0, 1)]), reason="empty choice"),
        NodeClosed(Path([("a", 0, 1)])),
        #
        NodeOpened(Path([("a", 1, 2)])),
        NodeOpened(Path([("a", 1, 2), ("b", 0, 1)])),
        Done(Path([("a", 1, 2), ("b", 0, 1)]), result=3),
        NodeClosed(Path([("a", 1, 2), ("b", 0, 1)])),
        NodeClosed(Path([("a", 1, 2)])),
        #
        NodeOpened(Path([("a", 2, 3)])),
        NodeOpened(Path([("a", 2, 3), ("b", 0, 1)])),
        Done(Path([("a", 2, 3), ("b", 0, 1)]), result=4),
        NodeClosed(Path([("a", 2, 3), ("b", 0, 1)])),
        #
        NodeOpened(Path([("a", 2, 3), ("b", 1, 2)])),
        Done(Path([("a", 2, 3), ("b", 1, 2)]), result=5),
        NodeClosed(Path([("a", 2, 3), ("b", 1, 2)])),
        NodeClosed(Path([("a", 2, 3)])),
        #
        NodeClosed(Path()),
    ]


def test_reports_node_events_thunk_values():
    def fun(amb):
        a = amb("a", [once(lambda: 3)])
        return a

    events = list(nondeterministic(fun, node_events=True))

    assert events == [
        NodeOpened(Path()),
        NodeOpened(Path([("a", 0, 3)])),
        Done(Path([("a", 0, 3)]), result=3),
        NodeClosed(Path([("a", 0, 3)])),
        NodeClosed(Path()),
    ]


def test_reports_node_events_failed_thunk(some_exception, mocker):
    bomb = mocker.MagicMock(side_effect=ValueError("boom"))

    def fun(amb):
        a = amb("a", [once(bomb)])
        return a

    events = list(nondeterministic(fun, node_events=True))

    ANY = mocker.ANY

    assert events == [
        NodeOpened(Path()),
        NodeOpened(Path([("a", 0, ANY)])),
        Failed(Path([("a", 0, ANY)]), exc=some_exception(ValueError, "boom")),
        NodeClosed(Path([("a", 0, ANY)])),
        NodeClosed(Path()),
    ]


def test_reports_node_events_failure_before_thunk(calls, some_exception, mocker):
    mock = mocker.MagicMock(return_value="foo")

    def fun(amb):
        if calls.count > 0:
            raise ValueError("Nope")

        a = amb("a", ["bar", once(mock)])
        calls.record(a=a)

    events = list(nondeterministic(fun, node_events=True))

    ANY = mocker.ANY

    assert events == [
        NodeOpened(Path()),
        #
        NodeOpened(Path([("a", 0, "bar")])),
        Done(Path([("a", 0, "bar")]), result=None),
        NodeClosed(Path([("a", 0, "bar")])),
        #
        NodeOpened(Path([("a", 1, ANY)])),
        Failed(Path([("a", 1, ANY)]), exc=some_exception(ValueError, "Nope")),
        NodeClosed(Path([("a", 1, ANY)])),
        #
        NodeClosed(Path()),
    ]


async def async_collect(async_gen):
    return [item async for item in async_gen]


@pytest.mark.asyncio()
async def test_async_reports_results():
    def fun(amb):
        a = amb("a", [1, 2])
        b = amb("b", [3, 4])
        return a + b

    events = await async_collect(nondeterministic(fun, mode="async"))

    assert events == [
        Done(path=Path([("a", 0, 1), ("b", 0, 3)]), result=4),
        Done(path=Path([("a", 0, 1), ("b", 1, 4)]), result=5),
        Done(path=Path([("a", 1, 2), ("b", 0, 3)]), result=5),
        Done(path=Path([("a", 1, 2), ("b", 1, 4)]), result=6),
    ]


# Since parallel executors by their nature cannot guarantee execution order, we test
# them on a series of example functions checking two things:
#
#   - is the event set the same as during the serial execution?
#   - does the event sequence make sense?
#
# The first one is easy to test: we compare sets of slightly adjusted events (to ensure
# they can be reasonably eq-compared). The second condition is more complicated, and is
# defined by `assert_events_make_sense` function below. It basically boils down to
# making sure nodes are opened, finished and closed in a proper order, children are
# processed before parents etc.


@dataclass
class Node:
    children: list[Node | None]
    parent: Node | None
    name: str
    value: object
    idx: int
    opened: bool = False
    finished: bool = False
    closed: bool = False


class EventsError(Exception):
    pass


def path_to_str(path: Path, mark: set[int] | None = None) -> str:
    if mark is None:
        mark = set()

    if not path:
        return "    (root node)\n"

    result = ""
    for j, (name, idx, value) in enumerate(path, start=1):
        fill = ">>> " if j - 1 in mark else "    "
        result += f"{fill}{j}. {name} [{idx}] = {value}\n"

    return result


def pad_with(items: list[Node | None], n: int, value: object) -> None:
    missing = n - len(items)
    if missing > 0:
        items += [None] * missing


def get_node(root: Node, event: Event) -> Node:
    node = root
    for i, c in enumerate(event.path):
        if c.idx >= len(node.children) or node.children[c.idx] is None:
            # new node
            if i == len(event.path) - 1:
                pad_with(node.children, c.idx + 1, None)
                new_node = Node([], node, c.name, c.value, c.idx)
                node.children[c.idx] = new_node
                path = path_to_str(event.path)
                prefix = f"New node with path:\n\n{path}\n"

                def fail(msg):
                    raise EventsError(prefix + msg)  # noqa: B023

                if not node.opened:
                    fail("parent not opened")
                if node.finished:
                    fail("parent finished")
                if node.closed:
                    fail("parent already closed")

            else:
                msg = (
                    f"Event:\n\n    {event}\n\n"
                    "has invalid path:\n\n"
                    f"{path_to_str(event.path, mark={i})}\n"
                )
                if c.idx >= len(node.children):
                    msg += f"Reason: index too large ({c.idx} >= {len(node.children)})"
                else:
                    msg += f"Reason: child with index {c.idx} does not exist yet"
                raise EventsError(msg)

        child = node.children[c.idx]
        assert child is not None
        node = child

        if node.value != c.value:
            msg = (
                f"Event:\n\n    {event}\n\n"
                "with path:\n\n"
                f"{path_to_str(event.path, mark={i})}\n"
                f"inconsistent value: {c.value} != {node.value}"
            )
            raise EventsError(msg)

    return node


def assert_events_make_sense(events: Sequence[Event]) -> None:
    __tracebackhide__ = True

    root = Node([], parent=None, name="", idx=0, value=None, opened=False, closed=False)

    error_msg = None
    try:
        for event in events:
            node = get_node(root, event)
            path = path_to_str(event.path)
            prefix = f"Node with path:\n\n{path}\n"

            def fail(msg):
                raise EventsError(prefix + msg)  # noqa: B023

            match event:
                case NodeOpened():
                    if node.opened:
                        fail("opened more than once")
                    if node.closed:
                        fail("opened after being closed")
                    if node.finished:
                        fail("opened after being finished")
                    node.opened = True

                case Finished():
                    if not node.opened:
                        fail("finished without being opened")
                    if node.finished:
                        fail("finished more than once")
                    if node.closed:
                        fail("finished after already closed")
                    if node.children:
                        fail("was finished, but has children")
                    node.finished = True

                case NodeClosed():
                    if not node.opened:
                        fail("closed without being opened")
                    if not node.children and not node.finished:
                        fail("closed as a leaf without being finished")
                    if node.closed:
                        fail("closed more than once")

                    problem = any(c is None or not c.closed for c in node.children)
                    if problem:
                        msg = "closed before children:\n"
                        for idx, c in enumerate(node.children):
                            if c is None:
                                msg += f"    [{idx}] None\n"
                            elif not c.closed:
                                msg += (
                                    f"    [{c.idx}] not closed "
                                    f"({c.name}, {c.value})\n"
                                )
                        fail(msg)
                    node.closed = True

    except EventsError as e:
        error_msg = e.args[0]

    if error_msg:
        pytest.fail(error_msg)


# To ensure events can be equality-tested, we replace problematic
# parts with fake placeholders.


@dataclass(frozen=True)
class ExceptionRepr:
    class_: type[Exception]
    message: str


@dataclass(frozen=True)
class ThunkRepr:
    pass


def normalize_choice(choice: Choice) -> Choice:
    if isinstance(choice.value, Thunk):
        thunk = choice.value
        return choice._replace(value=ThunkRepr())
    else:
        return choice


def normalize_path(path: Path) -> Path:
    return Path(map(normalize_choice, path))


def normalize_event(event: Event) -> Event:
    event = dataclasses.replace(event, path=normalize_path(event.path))

    match event:
        case Failed() as failed:
            exc_repr = ExceptionRepr(type(failed.exc), str(failed.exc))
            # Yes, this is not the correct type.
            # We deliberately bypass the typing system to ensure we can
            # test event equality in a sane way.
            return dataclasses.replace(failed, exc=exc_repr)  # type: ignore

        case _:
            return event


def normalize_events(events: Iterable[Event]) -> list[Event]:
    return [normalize_event(e) for e in events]


EXAMPLE_FUNCTIONS = []


def case(f):
    EXAMPLE_FUNCTIONS.append(f)
    return f


@case
def single_var_single_val(amb):
    return amb("val", [1])


@case
def single_var_many_vals(amb):
    return amb("val", [1, 2, 3])


@case
def many_vars_single_val(amb):
    a = amb("a", [1])
    b = amb("b", [2])
    return a + b


@case
def many_vars_many_vals(amb):
    a = amb("a", [1, 2, 3])
    b = amb("b", [3, 4, 5])
    return a + b


@case
def cartesian_prod_3d(amb):
    a = amb("a", range(4))
    b = amb("b", range(3))
    c = amb("c", range(2))
    return 100 * a + 10 * b + c


@case
def no_choice(amb):
    return 3


@case
def abort_no_choice(amb):
    amb.abort()


@case
def abort_unconditional(amb):
    a = amb("a", [1, 2, 3])
    b = amb("b", [4, 5, 6])
    amb.abort()


@case
def abort_conditional(amb):
    a = amb("a", [1, 2, 3])
    b = amb("b", [4, 5, 6])
    amb.ensure(a + b < 8)


@case
def path_dependent_value(amb):
    a = amb("a", [1, 2, 3, 4])
    if a % 2 == 0:  # noqa: SIM108
        b = amb("b", ["foo", "bar"])
    else:
        b = amb("b", ["aaa", "bbb"])


@case
def path_dependent_name(amb):
    a = amb("a", [1, 2, 3, 4])
    if a % 2 == 0:
        b = amb("b", ["foo", "bar"])
    else:
        c = amb("c", ["aaa", "bbb"])


@case
def path_dependent_shape(amb):
    a = amb("a", [1, 2, 3, 4])
    if a % 2 == 0:
        b = amb("b", ["foo", "bar"])
        c = amb("c", [10, 20])
    else:
        return 3


def no_blowup(a: int) -> int:
    return 3 * a


def blowup(a: int) -> int:
    if a == 1:
        raise ValueError("boom")
    return a + 2


@case
def complex_logic(amb):
    a = amb("a", [1, 2, 3, 4])
    if a % 2 == 0:
        b = amb("b", ["foo", "bar", "lol"])
        match b:
            case "foo" | "bar":
                c = amb("c", [a + 1, b])
                d = amb("d", ["a", 3 if a > 2 else 4])
                amb.ensure(d + a < 7)
                if b == "bar":
                    e = amb("e", [5, 8])
                    return d * e
                else:
                    return a + 3 * d
            case _:
                d = amb("d", [2, 4, 8])
                m = amb("m", [7, once(blowup), once(no_blowup)])
                if a + d > 7:
                    amb.abort()
                return a - d
    else:
        c = amb("c", range(a))
        h = amb("h", [c - 1, c + 1])
        amb.ensure(h > 0)
        if a == 4:
            g = amb("g", ["foo", "baaar"])
            return h * len(g)
        else:
            amb.abort()


def make_inner():
    def inner(amb):
        return amb("ret", [1, 2, 3])

    return inner


def make_bound_method():
    class Foo:
        def bar(self, amb):
            return 3

    foo = Foo()
    return foo.bar


EXAMPLE_FUNCTIONS += [
    pytest.param(lambda amb: amb("ret", [1, 2, 3]), id="lambda"),
    pytest.param(make_inner(), id="local function"),
    pytest.param(make_bound_method(), id="bound method"),
]


# we want fullest possible event log
ALL_EVENTS = dict(node_events=True, aborted=True)


def serial(fun) -> list[Event]:
    events = nondeterministic(fun, execution="serial", mode="sync", **ALL_EVENTS)
    return normalize_events(events)


@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
def test_serial(fun):
    events = serial(fun)
    assert_events_make_sense(events)


@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
@pytest.mark.asyncio()
async def test_async_serial(fun):
    serial_events = serial(fun)
    events = await async_collect(nondeterministic(fun, mode="async", **ALL_EVENTS))
    assert normalize_events(events) == serial_events


@pytest.mark.timeout(1)
@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
@pytest.mark.threads()
def test_threads(fun):
    serial_events = serial(fun)

    events = nondeterministic(fun, mode="sync", execution="threads", **ALL_EVENTS)
    events = list(events)

    assert_events_make_sense(events)
    normalized = normalize_events(events)
    assert set(normalized) == set(serial_events)


@pytest.mark.timeout(1)
@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
@pytest.mark.threads()
@pytest.mark.asyncio()
async def test_async_threads(fun):
    serial_events = serial(fun)

    events = await async_collect(
        nondeterministic(fun, mode="async", execution="threads", **ALL_EVENTS)
    )

    assert_events_make_sense(events)
    normalized = normalize_events(events)
    assert set(normalized) == set(serial_events)


@pytest.mark.slow()
@pytest.mark.timeout(1)
@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
@pytest.mark.processes()
def test_processes(fun):
    serial_events = serial(fun)

    events = nondeterministic(fun, execution="processes", **ALL_EVENTS)
    events = list(events)

    assert_events_make_sense(events)
    normalized = normalize_events(events)
    assert set(normalized) == set(serial_events)


@pytest.mark.slow()
@pytest.mark.timeout(1)
@pytest.mark.parametrize("fun", EXAMPLE_FUNCTIONS)
@pytest.mark.processes()
@pytest.mark.asyncio()
async def test_async_processes(fun):
    serial_events = serial(fun)

    events = await async_collect(
        nondeterministic(fun, mode="async", execution="processes", **ALL_EVENTS)
    )

    assert_events_make_sense(events)
    normalized = normalize_events(events)
    assert set(normalized) == set(serial_events)


@pytest.mark.slow()
@pytest.mark.timeout(1)
def test_processes_no_dill():
    try:
        fun = complex_logic
        events = nondeterministic(fun, execution="processes", use_dill=False)
        events = list(events)
    except Exception:
        pytest.fail("Unexpected failure")


def test_environment_implements_execution_environment():
    from supy.amb import _Environment, _SerialExecutor

    env = _Environment(_SerialExecutor(), lambda: None)
    assert isinstance(env, ExecutionEnvironment)
