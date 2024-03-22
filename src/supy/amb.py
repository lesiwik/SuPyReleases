"""Support for non-deterministic function execution."""

# Implementation overview:
#
# Each invocation of Amb.__call__ is a decision that needs to be made, a branching
# point. Execution can follow different paths from this point on, depending on which
# value is returned to the caller. The execution process is internally represented as a
# tree with a single root, and interior nodes corresponding to the choices made during
# the execution. Initially this tree only has a root. Subsequent executions of the
# non-deterministic function reveal the choices and create new nodes, expanding the
# decision tree.
#
# Each invocation of the non-deterministic function is associated with a certain node in
# this tree. The choices leading up to and including this node are replayed. After that,
# if new choices are encountered, they are added to the tree as new nodes. The current
# invocation always takes the first value from the list of options, while other
# possibilities are queued for later exploration.
#
# In terms of code, the control flow looks like this:
#
# nondeterministic() called:
# - creates an instance of _NondeterministicExecutor
# - calls its run() / run_async() method
#
# _NondeterministicExecutor:
# - pushes the root of decision tree onto the node stack
# - in the loop:
#   - pops a node from the node stack and schedule the associated execution path using
#     the _execute_task function
#   - handles messages from tasks
#   - generates events for the caller of nondeterministic()
#   For more details about the loop, see comments in _NondeterministicExecutor.run()
#
# _Task execution:
# - sends _TaskStarted message to the executor
# - calls the non-deterministic function with _Context object, which contains all the
#   choices made on the path leading to the node from which the task was made
# - for every choice already made, _Context object replays the choice when a value is
#   requested
# - when a new choice is encountered, sends _TaskBranched message to the executor
# - after the non-deterministic function call is completed, sends _TaskFinished message
#   to the executor
#
# Details regarding Thunk handling have been omitted.

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import itertools
import multiprocessing
import queue
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterable, Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import (
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import dill


class Amb(ABC):
    """Interface of the non-deterministic execution control."""

    @abstractmethod
    def __call__(self, name: str, options: Iterable[object]) -> object:
        """
        Non-deterministically choose one value from the list.

        Parameters
        ----------
        name : str
            Name of the variable whose value is being chosen. It is used in descriptions
            of the execution path attached to events generated during the execution.
        options : iterable object
            Sequence of values to choose from. If empty, the computation is aborted and
            an `EmptyChoice` event is generated to signal it.

        Returns
        -------
        object
            One of the values passed in `options`.

        See Also
        --------
        nondeterministic : Executes a non-deterministic computation.
        """

    def abort(self, message: str | None = None) -> None:
        """
        Abort the execution path.

        Aborting an execution path is considered to be a normal control structure in the
        context of non-deterministic functions. Its intended use is to terminate the
        exploration of some parameter space upon reaching a state where some constraints
        are violated (see examples in `nondeterministic`). Aborting an execution path is
        not considered an error, and should not be used to signal it. For that purpose,
        use regular Python exceptions.

        Parameters
        ----------
        message : str, optional
            Optional reason for aborting the computation.
        """
        raise _Abort(message)

    def ensure(self, cond: object, message: str | None = None) -> None:
        """
        Abort the execution path unless the condition is satisfied.

        Parameters
        ----------
        cond : bool
            Whether to abort the computation. Any value that can be interpreted as bool
            can be used (iterables, types with `__bool__`).
        message : str, optional
            Optional reason for aborting the computation.
        """
        if not cond:
            self.abort(message)

    @abstractmethod
    def past_choices(self) -> Path:
        """
        List choices made during the current execution.

        Each call to the `Amb` object is represented by a single `Choice` object
        recording the name, value, and index in the list of choices.

        Returns
        -------
        Path
            Tuple of `Choice` objects describing the current execution path leading up
            to this point.
        """
        ...


AmbFun: TypeAlias = Callable[[Amb], object]
"""
Type of non-deterministic computations.
"""


class Error(Exception):
    """
    Base class for non-deterministic execution exceptions.

    Exceptions deriving from this class are used internally in the implementation of
    the non-deterministic execution and should not be caught nor raised by users.
    """


class _ThunkEvalError(Error):
    """Wraps the exception from thunk evaluation."""


@dataclass
class _Abort(Error):
    """Raised to exit the non-deterministic function."""

    reason: str | None = None


class _AbortFromEmptyChoice(_Abort):
    """Raised when the list of choices is empty."""

    def __init__(self):
        super().__init__("empty choice")


class Choice(NamedTuple):
    """
    A single choice made on the execution path.

    Attributes
    ----------
    name : str
        Name of the requested non-deterministic value.
    idx : int
        Index of the value chosen from the list of options.
    value : object
        Value chosen from the list of options.
    """

    name: str
    idx: int
    value: object


Path: TypeAlias = tuple[Choice, ...]
"""
A sequence of `Choice` objects specifying a path in the decision tree.
"""


@dataclass(frozen=True)
class Event:
    """
    Base class of events occurring during non-deterministic execution.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    """

    path: Path


@dataclass(frozen=True)
class Finished(Event):
    """
    Event denoting that an execution path has finished.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    """


@dataclass(frozen=True)
class Done(Finished):
    """
    Event denoting that an execution path has finished successfully.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    result : object
        Value returned by the computation.
    """

    result: object


@dataclass(frozen=True)
class Failed(Finished):
    """
    Event denoting that an execution path has encountered an exception.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    exc : Exception
        Exception that terminated this execution path.
    """

    exc: Exception


@dataclass(frozen=True)
class Aborted(Finished):
    """
    Event denoting that an execution path has been aborted.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    reason : str, optional
        Message describing why the path was terminated.
    """

    reason: str | None = None


@dataclass(frozen=True)
class EmptyChoice(Aborted):
    """
    Event denoting that an execution path has encountered an empty choice.

    This results in the computation being aborted.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    reason : str, optional
        Cause of abortion, for this class it is always ``"empty choice"``.
    """

    reason: str = "empty choice"


@dataclass(frozen=True)
class NodeEvent(Event):
    """
    Base class of events related to nodes of the decision tree.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    """


@dataclass(frozen=True)
class NodeOpened(NodeEvent):
    """
    Event denoting the start of exploration of the subtree at this node.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    """


@dataclass(frozen=True)
class NodeClosed(NodeEvent):
    """
    Event denoting the end of exploration of the subtree at this node.

    Attributes
    ----------
    path : Path
        Path to the node of the decision tree involved in the represented event.
    """


T = TypeVar("T")


class Thunk(Generic[T]):
    """
    Wraps a callable producing a value to ensure it is executed only once.

    The intended use of this class is to wrap expensive computations whose results
    should be shared between non-deterministic execution paths. Even though only the
    options passed to `Amb` during the first execution path leading through any given
    node of the decision tree are used and the values passed during execution of
    subsequent paths are ignored, they are still evaluated, since Python's evaluation
    model is eager. To prevent that, we can wrap a computation (i.e. a callable
    accepting no arguments) in an instance of this class and pass it to `Amb` as on of
    the choices. When the non-deterministic execution system encounters a `Thunk` value,
    it evaluates it before returning it to the calling function and saves the result to
    be used for subsequent calls at this node. As a result, the wrapped function is
    called exactly once.

    Parameters
    ----------
    func : callable
        A callable accepting no argument and producing a value of type ``T``.

    See Also
    --------
    once : Wraps a function together with arguments.

    Notes
    -----
    `Thunk` objects do not appear as choice values in paths, unless an exception occurs
    during its evaluation, or before it on the first execution path that would lead
    through the associated node.
    """

    def __init__(self, func: Callable[[], T]):
        self._func = func

    def __call__(self) -> T:
        """
        Execute the wrapped computation and return the produced value.

        Returns
        -------
        object
            Value produced by the wrapped computation.
        """
        return self._func()


P = ParamSpec("P")


def once(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Thunk[T]:
    """
    Wrap function and its arguments into a `Thunk` object.

    Parameters
    ----------
    func : callable
        A callable accepting the arguments that follow.
    *args : tuple, optional
        Positional arguments.
    **kwargs : dict, optional
        Keyword arguments.

    Returns
    -------
    Thunk
        `Thunk` object wrapping the computation consisting of applying `func` to `args`
        and `kwargs`.

    Examples
    --------
    >>> thunk = once(max, [3, 5, 2, 4, 1, 8, 5, 6])
    >>> thunk()
    8

    >>> thunk = once(str.format, "{} + {} = {result}", 1, 3, result=4)
    >>> thunk()
    '1 + 3 = 4'
    """
    return Thunk(functools.partial(func, *args, **kwargs))


@dataclass(kw_only=True)
class _Node:
    """Base class of nodes of the decision tree."""

    children: int = 1
    finished: int = 0
    parent: _Node | None = None

    def expand(self, count: int) -> None:
        self.children = count

    def update(self) -> bool:
        self.finished += 1
        return self.done

    @property
    def done(self) -> bool:
        return self.finished == self.children


@dataclass
class _RootNode(_Node):
    pass


@dataclass
class _InnerNode(_Node):
    choice: Choice


def _from_root_to_node(node: _Node) -> list[_Node]:
    path = []

    n: _Node | None = node
    while n is not None:
        path.append(n)
        n = n.parent

    path.reverse()
    return path


def _choices_along(tree_path: list[_Node]) -> Path:
    # first node is root
    inner = typing.cast(list[_InnerNode], tree_path[1:])
    return Path(node.choice for node in inner)


def _choices_up_to(node: _Node) -> Path:
    path = _from_root_to_node(node)
    return _choices_along(path)


class _Context(Amb):
    """
    Convrete implementation of `Amb`.

    Each non-deterministic execution path uses a single instance of this class. It is
    instantiated at a specific node of the decision tree, and as the execution goes past
    it, it drills down into the leftmost path, notifying the executor about the branches
    along the way, through the message queue.
    """

    def __init__(self, past_choices: Path, msg_queue: _MessageQueue):
        self.path = list(past_choices)  # new choices are appended as we go
        self.ptr = -1
        self.msg_queue = msg_queue

    def __call__(self, name: str, options: Iterable[object]) -> object:
        self.move_down()

        if not self.has_next:
            self.make_next_choice(name, list(options))

        return self.current_choice_value()

    def make_next_choice(self, name: str, options: list[object]) -> None:
        if not options:
            raise _AbortFromEmptyChoice()

        choice = Choice(name, 0, options[0])
        self.path.append(choice)

        self.msg_queue.send(_TaskBranched, name, options)

    def current_choice_value(self) -> object:
        entry = self.current_choice.value

        if isinstance(entry, Thunk):
            assert self.ptr == self.depth - 1, "can only happen at the tip"
            try:
                value = entry()
                new_entry = self.current_choice._replace(value=value)
                self.path[self.ptr] = new_entry
                self.msg_queue.send(_ThunkComputed, value)
            except Exception as e:
                raise _ThunkEvalError() from e

        return self.current_choice.value

    def move_down(self):
        self.ptr += 1

    @property
    def has_next(self) -> bool:
        return self.ptr < self.depth

    @property
    def current_choice(self) -> Choice:
        return self.path[self.ptr]

    @property
    def depth(self) -> int:
        return len(self.path)

    def past_choices(self) -> Path:
        return Path(self.path)


class _MessageQueue:
    def __init__(self, task_id: int, msg_queue):
        self.task_id = task_id
        self.msg_queue = msg_queue

    def send(self, msg_type: type[_Message], *args, **kwargs) -> None:
        """
        Send a message build from type and argument list.

        Argument `task_id` is added automatically and should not be passed to `send`.
        """
        msg = msg_type(*args, **kwargs, task_id=self.task_id)
        self.msg_queue.put(msg)


@dataclass
class _Task:
    """Structure representing a single execution path."""

    fun: AmbFun
    ctx: _Context
    task_id: int


@dataclass(kw_only=True)
class _Message:
    """Base class of messages sent by tasks to the executor."""

    task_id: int


@dataclass
class _TaskStarted(_Message):
    """Sent by each task before executing the payload."""


@dataclass
class _TaskFinished(_Message):
    """Sent by each task after executing the payload (success or not)."""

    result: Finished


@dataclass
class _TaskBranched(_Message):
    """Sent by `_Context` upon reaching a new branching in the decision tree."""

    name: str
    choices: list[object]


@dataclass
class _ThunkComputed(_Message):
    """Sent by `_Context` after evaluating a computation wrapped in a `Thunk`."""

    value: object


def _aborted_from(path: Path, e: _Abort) -> Aborted:
    # we distinguish between aborts explicitly requested by the function,
    # and those caused by an empty choice list
    match e:
        case _AbortFromEmptyChoice():
            return EmptyChoice(path)
        case _:
            return Aborted(path, e.reason)


def _result_of_task(task: _Task) -> Finished:
    """Execute the task payload and wrap the result."""
    ctx = task.ctx
    try:
        val = task.fun(ctx)
        return Done(ctx.past_choices(), val)
    except _Abort as e:
        return _aborted_from(ctx.past_choices(), e)
    except _ThunkEvalError as e:
        # it is always raised from Exception
        original = typing.cast(Exception, e.__cause__)
        return Failed(ctx.past_choices(), original)
    except Exception as e:
        return Failed(ctx.past_choices(), e)


def _execute_task(task: _Task) -> None:
    """Full task execution routine, as submitted to the execution environment."""
    task.ctx.msg_queue.send(_TaskStarted)
    result = _result_of_task(task)
    task.ctx.msg_queue.send(_TaskFinished, result)


def _format_exception_msg(e: Exception) -> str:
    import traceback

    lines = traceback.format_exception(e)
    return "".join(lines)


def _warn_about_exception(e: Exception, ctx: _Context, stacklevel: int) -> None:
    msg = "Exception during executing non-deterministic runner task:\n"
    msg += _format_exception_msg(e)

    path = ctx.past_choices()
    if path:
        msg += "on path:\n"
        for name, idx, value in ctx.past_choices():
            msg += f"  [{idx}] {name} = {value!r}\n"
    else:
        msg += "in the root node.\n"

    explanation = """
    The most likely reason is that the process-based parallel executor was
    used, but the function to be executed does not respect the limitations
    imposed by Python's `multiprocessing` module used to implement the
    process-based executor.

    In particular,

      - non-deterministic values used in the function
      - return value
      - function itself

    must be picklable. This excludes lambda functions and bound methods,
    among other things. For a full list, please consult:

    https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    """

    import textwrap

    msg += textwrap.dedent(explanation)
    warnings.warn(msg, stacklevel=stacklevel)


def _future_done_callback(future, task: _Task) -> None:
    """
    Log task exceptions encountered *before* actually executing the task.

    Exceptions during the task payload execution are wrapped into a `Failed` object and
    returned as the result of the execution path. The only other parts that may fail are
    communication with the executor, and the execution environment itself. So far, the
    only circumstances in which such a failure has been encountered were issues related
    to pickling task objects by the `concurrent.futures.ProcessPoolExecutor`.

    This function should be attached to futures returned by the task submission
    function.
    """
    if (exc := future.exception()) is not None:
        _warn_about_exception(exc, task.ctx, stacklevel=2)

        import logging

        logging.basicConfig()
        logger = logging.getLogger(__name__)
        msg = (
            "Exception during execution of the non-deterministic runner task. "
            "The most likely reason is that the process-based parallel executor "
            "is used, but the function to be executed does not respect the "
            "limitations imposed by Python's `multiprocessing` module used "
            "to implement the process-based executor. In particular, all the "
            "non-deterministic values, function return value and the function "
            "itself must be picklable. This excludes lambda functions and "
            "bound methods, among other things. For a full list, please consult "
            "https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled"
        )
        logger.critical(msg, exc_info=exc)


def _has_thunk(node: _Node) -> bool:
    return isinstance(node, _InnerNode) and isinstance(node.choice.value, Thunk)


def _with_path_from(event: Event, node: _Node) -> Event:
    """Replace the event path with the one leading to given node."""
    path = _choices_up_to(node)
    return dataclasses.replace(event, path=path)


class _ChoiceTree:
    """Helper class managing node-related events."""

    def __init__(self, event_queue: deque[Event]) -> None:
        self.event_queue = event_queue
        self.root = _RootNode()

    def open_node(self, node: _Node) -> None:
        if not _has_thunk(node):
            # In case of a thunk, defer the event until we have a computed value
            # that can be used in the path.
            self.post_node_opened_event(node)

    def post_node_opened_event(self, node: _Node) -> None:
        path = _choices_up_to(node)
        self.event_queue.append(NodeOpened(path))

    def close_node(self, node: _Node) -> None:
        while node.update():
            path = _choices_up_to(node)
            self.event_queue.append(NodeClosed(path))

            if node.parent is not None:
                node = node.parent
            else:
                break

    def branch(self, node: _Node, name: str, choices: list[object]) -> list[_InnerNode]:
        children = [
            _InnerNode(Choice(name, i, val), parent=node)
            for i, val in enumerate(choices)
        ]
        node.expand(len(children))
        return children

    def update_value(self, node: _Node, value: object) -> None:
        assert isinstance(node, _InnerNode), "root has no value"
        node.choice = node.choice._replace(value=value)
        self.post_node_opened_event(node)

    def node_failed(self, node: _Node) -> None:
        if _has_thunk(node):
            # Computation has failed before or during the thunk evaluation.
            # Thunk will never be evaluated, do not defer the event any longer.
            self.post_node_opened_event(node)

    @property
    def done(self) -> bool:
        return self.root.done


class _TaskManager:
    """Helper class managing the task state data."""

    def __init__(self, msg_queue) -> None:
        self.id_stream = itertools.count()
        self.id_to_node: dict[int, _Node] = {}
        self.msg_queue = msg_queue

    def next_task_id(self) -> int:
        return next(self.id_stream)

    def create(self, fun: AmbFun, node: _Node) -> _Task:
        task_id = self.next_task_id()
        self.id_to_node[task_id] = node

        choices = _choices_up_to(node)
        msg_queue = _MessageQueue(task_id, self.msg_queue)

        ctx = _Context(choices, msg_queue)
        return _Task(fun, ctx, task_id)

    def node(self, task_id: int) -> _Node:
        return self.id_to_node[task_id]

    def finished(self, task_id: int) -> None:
        # delete the reference to the node so that it can be promptly
        # garbage collected
        del self.id_to_node[task_id]

    def update_node(self, task_id: int, node: _Node) -> None:
        self.id_to_node[task_id] = node


class _NondeterministicExecutor:
    def __init__(
        self,
        environment,
        *,
        event_filter: Callable[[Event], bool],
        pass_exceptions: bool,
    ):
        self.environment = environment
        self.event_filter = event_filter
        self.pass_exceptions = pass_exceptions

    def setup(self) -> None:
        # All the state is recreated for each run() call
        # Only the configuration is persistent
        self.msg_queue = self.environment.make_queue()
        self.event_queue: deque[Event] = deque()
        self.node_stack: deque[_Node] = deque()

        self.tasks = _TaskManager(self.msg_queue)
        self.tree = _ChoiceTree(self.event_queue)

        self.node_stack.append(self.tree.root)

    def flush_event_queue(self):
        while self.event_queue:
            event = self.event_queue.popleft()
            if self.event_filter(event):
                yield event

    def handle_possible_failure(self, finished: _TaskFinished) -> None:
        match finished.result:
            case Failed() as f:
                # In presence of thunks, the NodeOpened event might not have been
                # generated yet. We need to give the tree event manager a chance
                # to do it before reraising the exception.
                node = self.tasks.node(finished.task_id)
                self.tree.node_failed(node)

                if self.pass_exceptions:
                    raise f.exc

    def schedule_task(self, task: _Task) -> None:
        future = self.environment.submit(_execute_task, task)
        if future is not None:
            callback = functools.partial(_future_done_callback, task=task)
            future.add_done_callback(callback)

    def schedule_node(self, fun: AmbFun, node: _Node) -> None:
        task = self.tasks.create(fun, node)
        self.schedule_task(task)

    def schedule_next_node(self, fun: AmbFun) -> None:
        node = self.node_stack.pop()
        self.schedule_node(fun, node)

    def handle_message(self, msg: _Message) -> None:
        node = self.tasks.node(msg.task_id)

        match msg:
            case _TaskStarted():
                self.tree.open_node(node)

            case _TaskFinished() as finished:
                self.handle_possible_failure(finished)
                # In case of thunk failure, this replaces possibly remote thunk copy
                # with the local object, ensuring consistent object identity in paths
                result = _with_path_from(finished.result, node)
                self.event_queue.append(result)
                self.tree.close_node(node)
                self.tasks.finished(msg.task_id)

            case _TaskBranched() as branching:
                children = self.tree.branch(node, branching.name, branching.choices)
                first, *rest = children
                # The first node is already handled by the task that sent this message.
                self.tree.open_node(first)
                self.tasks.update_node(msg.task_id, first)
                # Remaining nodes are pushed onto the node stack in reverse order to
                # ensure standard backtracking exploration order during serial
                # execution.
                self.node_stack.extend(reversed(rest))

            case _ThunkComputed() as computed:
                self.tree.update_value(node, computed.value)

    def process_message(self, msg: _Message) -> Iterable[Event]:
        # Handle the message and push out generated events
        self.handle_message(msg)
        yield from self.flush_event_queue()

    def process_next_message(self, *, block: bool) -> Iterable[Event]:
        # Handle the next message and push out generated events
        msg = self.msg_queue.get(block=block)
        yield from self.process_message(msg)

    def process_all_available_msgs(self) -> Iterable[Event]:
        # Handle all the messages currently in the queue
        with contextlib.suppress(queue.Empty):
            while True:
                yield from self.process_next_message(block=False)

    async def async_get_message(self) -> _Message:
        return await asyncio.to_thread(lambda: self.msg_queue.get(block=True))

    def run(self, fun: AmbFun) -> Iterable[Event]:
        self.setup()

        with self.environment:
            # The somewhat bizarre construction of this event loop ensures that paths
            # are executed in an order consistent with a simple backtracking
            # implementation during serial execution, and are scheduled in an order
            # close to that during parallel execution.
            #
            # Since decision tree nodes are created in response to messages, the way in
            # which path execution and message handling is interlaced does impact the
            # order in serial execution. The desired order is depth-first, but we still
            # want to process messages in chronological order, so we use a stack of
            # nodes instead of a queue, and we eagerly process messages as soon as
            # possible - messages from each execution path are processed before the next
            # node is picked. This ensures that the next node we pick for execution will
            # be the one discovered the latest, which is exactly what we want.
            #
            # During serial execution, schedule_next_node actually performs the whole
            # computation, so all the messages produced during its execution are already
            # in the message queue after its return. For parallel execution, however,
            # there is no such guarantee - after the inner while loop we may be left in
            # the situation where the node stack is empty, but there are, or eventually
            # will be new messages in the message queue, produced by the tasks currently
            # running. At this point, we need to wait for such messages, unless we know
            # for sure all the tasks are completed.
            while not self.tree.done:
                while self.node_stack:
                    self.schedule_next_node(fun)
                    yield from self.process_all_available_msgs()

                if not self.tree.done:
                    yield from self.process_next_message(block=True)

    async def async_run(self, fun: AmbFun) -> AsyncIterable[Event]:
        self.setup()

        with self.environment:
            # consult comments in run() about the loop structure
            while not self.tree.done:
                while self.node_stack:
                    self.schedule_next_node(fun)
                    for event in self.process_all_available_msgs():
                        yield event
                    await asyncio.sleep(0)

                if not self.tree.done:
                    msg = await self.async_get_message()
                    for event in self.process_message(msg):
                        yield event


@runtime_checkable
class ExecutionEnvironment(AbstractContextManager, Protocol):
    """
    Infrastructure for execution of non-deterministic functions.

    **Custom execution environments**

    Custom implementations of this protocol can be passed as `execution` argument to the
    `nondeterministic` function. The queue created by `make_queue` must support the
    interface of `queue.Queue`, and it must be possible to share between the main
    program and the tasks scheduled for execution using `submit`. Specifically, the
    tasks must be able to put items into it, and the main program must be able to get
    items from it.
    """

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs):
        """
        Submit a task for execution.

        Parameters
        ----------
        fn : callable
            Function to be executed.
        *args : tuple, optional
            Positional arguments to pass to `fn`.
        **kwargs : dict, optional
            Keyword arguments to pass to `fn`.

        Returns
        -------
        future object or None
            Object exposing `concurrent.futures.Future` interface.
        """

    def make_queue(self):
        """
        Create a message queue that can be passed to submitted tasks.

        Returns
        -------
        Queue
            Object supporting `queue.Queue` interface.
        """


class _Environment:
    """Execution environment wrapping a concrete executor."""

    def __init__(self, executor, make_queue):
        self.executor = executor
        self.make_queue = make_queue

    def submit(self, fn, /, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def __enter__(self):
        self.executor.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self.executor.__exit__(exc_type, exc_value, traceback)


class _SerialExecutor:
    """Trivial executor implementation."""

    def submit(self, fn, /, *args, **kwargs):
        fn(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _serial_execution(**kwargs) -> _Environment:
    executor = _SerialExecutor()
    return _Environment(executor, make_queue=Queue)


def _threads_execution(**kwargs) -> _Environment:
    executor = ThreadPoolExecutor(**kwargs)
    return _Environment(executor, make_queue=Queue)


def _processes_execcution(use_dill=True, **kwargs) -> _Environment:
    if use_dill:
        return _make_dill_processes_env(**kwargs)
    else:
        return _make_default_processes_env(**kwargs)


def _make_default_processes_env(**kwargs) -> _Environment:
    executor = ProcessPoolExecutor(**kwargs)
    manager = multiprocessing.Manager()
    return _Environment(executor, manager.Queue)


def _make_dill_processes_env(**kwargs) -> _Environment:
    executor = _DillExecutor(**kwargs)
    return _Environment(executor, _DillQueue)


# To sidestep limitations of Python's `multiprocessing` pickle-based serializer, we
# utilize dill. While it is possible to hack `multiprocessing` to use dill directly
# (https://stackoverflow.com/a/69061131/2725031), it feels rather brittle, since it
# touches the possibly unstable internals, not just the public API. Instead, we
# introduce a couple wrappers to make sure the data being sent between processes is
# already serialized using dill, so that all `pickle` sees are raw bytes.
#
# Data is sent between processes in two situations:
#
# - New task, function + args is sent for execution: this is handled by serializing
#   function and the arguments, and submitting a deserializing wrapper instead of the
#   original function (_DillExecutor)
#
# - Tasks sends messages back to the executor: this is handled by using a Queue wrapper
#   that serializes messages before `put`, and deserializes after `get` (_DillQueue)


def _dill_wrapper(fn_data, args_data, kwargs_data):
    fn_orig = dill.loads(fn_data)
    args_orig = dill.loads(args_data)
    kwargs_orig = dill.loads(kwargs_data)
    # return value does not need to be serialized, as _execute_task returns None
    return fn_orig(*args_orig, **kwargs_orig)


class _DillExecutor:
    def __init__(self, **kwargs):
        self.executor = ProcessPoolExecutor(**kwargs)

    def submit(self, fn, /, *args, **kwargs):
        fn_data = dill.dumps(fn)
        args_data = dill.dumps(args)
        kwargs_data = dill.dumps(kwargs)
        return self.executor.submit(_dill_wrapper, fn_data, args_data, kwargs_data)

    def __enter__(self):
        return self.executor.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self.executor.__exit__(exc_type, exc_value, traceback)


class _DillQueue:
    def __init__(self):
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()

    # For now, we only need these two Queue functions.
    # If this ever changes, this wrapper needs to be extended.

    def put(self, item: object) -> None:
        item_data = dill.dumps(item)
        self.queue.put(item_data)

    def get(self, block=True):
        item_data = self.queue.get(block)
        item = dill.loads(item_data)
        return item


@dataclass
class _EventFilter:
    """Controls which events are forwarded to the caller."""

    node_events: bool
    aborted: bool

    def __call__(self, event: Event) -> bool:
        match event:
            case NodeEvent():
                return self.node_events

            case Aborted():
                return self.aborted

            case _:
                return True


class Mode(Enum):
    """Event generation mode."""

    SYNC = 1
    ASYNC = 2


class Execution(Enum):
    """Execution mode."""

    SERIAL = 1
    THREADS = 2
    PROCESSES = 3


def _choose_environment(
    execution: str | Execution | ExecutionEnvironment, **kwargs
) -> ExecutionEnvironment:
    match execution:
        case ExecutionEnvironment():
            return execution
        case "serial" | Execution.SERIAL:
            return _serial_execution(**kwargs)
        case "threads" | Execution.THREADS:
            return _threads_execution(**kwargs)
        case "processes" | Execution.PROCESSES:
            return _processes_execcution(**kwargs)
        case _:
            raise ValueError(f"Unknown execution type '{execution}'")


def nondeterministic(
    fun: AmbFun,
    execution: Literal["serial", "threads", "processes"]
    | Execution
    | ExecutionEnvironment = "serial",
    mode: Literal["sync", "async"] | Mode = "sync",
    *,
    node_events: bool = False,
    aborted: bool = False,
    pass_exceptions: bool = False,
    **kwargs,
):
    """
    Execute a non-deterministic function.

    This function calls the supplied computation with a single argument of type `Amb`,
    which can be used from within the function body to retrieve non-deterministic values
    and abort execution if desired. Computation is invoked with every encountered
    combination of non-deterministic values. In case of serial execution, the choice
    tree generated during the computation is guaranteed to be explored in a depth-first
    search order and execution can be thought of as a backtracking process. The function
    returns an `Iterator` of `Event` objects describing the progress of exploration of
    available execution paths, or `AsyncIterator` in async mode. Exceptions during the
    execution by default are yielded as `Failed` events.

    Parameters
    ----------
    fun : callable
        Non-deterministic function to execute. It must accept a single `Amb` argument
        and can return value of any type.

    execution : {str, `Execution`, `ExecutionEnvironment`}, optional
        Available execution modes:

        - 'serial': Serial execution, guaranteed depth-first search order.
        - 'threads': Parallel execution using multiple threads.
        - 'processes': parallel execution using multiple processes.

        In case of 'threads' and 'processes' execution modes, additional arguments
        supplied as `kwargs` will be passed to the constructor of
        `concurrent.futures.ThreadPoolExecutor` or
        `concurrent.futures.ProcessPoolExecutor` used to schedule the computation paths.

        .. note::
            Due to Python's Global Interpreter Lock (GIL), only one thread can execute
            Python code at the same time. This means that for CPU-intensive tasks,
            thread-based parallelism will not give a significant performance improvement
            (unless the calculations are done by calling a native, compiled code).

        Custom execution method can be specified by passing an argument implementing the
        `ExecutionEnvironment` protocol.

    mode : {'sync', 'async', Mode}, optional
        Whether the stream of results is generated in a standard, or asynchronous mode.
        In a synchronous mode the function returns an `Iterator`, in the asynchronous
        mode - `AsyncIterator`.

    node_events : bool, optional
        If ``True``, in addition to `Finished` events signifying a computation path that
        reached its end, `NodeEvent` objects representing events associated with all the
        nodes of the computation tree (not only leaves) will be generated.

    aborted : bool, optional
        If ``False``, `Aborted` events signifying a computation path that was aborted
        will not be emitted. If `node_events` is ``True``, corresponding `NodeOpened`
        and `NodeClosed` events will still be present, resulting in a node without a
        termination event.

    pass_exceptions : bool, optional
        If ``True``, exceptions encountered during execution of computation paths will
        be propagated outside. The corresponding `Failed` event will not be produced.

    **kwargs : dict, optional
        Extra arguments passed to the constructor of the underlying task executor
        (`concurrent.futures.ThreadPoolExecutor` or
        `concurrent.futures.ProcessPoolExecutor`). See [2]_ for a list of available
        options.

    Yields
    ------
    Finished
        object containing the value produced by a single execution path of the
        computation, or an exception that terminated it. If `aborted` is ``True``, it
        may also be an instance of `Aborted`, with an optional termination reason.
    NodeEvent
        In case `node_events` is ``True``.

    Other Parameters
    ----------------
    use_dill : bool, default: True
        Only used with 'processes' execution mode.  If True, use `dill` instead of the
        default `multiprocessing` pickler to serialize data.

        .. warning::
            If `use_dill` is ``False``, the default serializer, based on Python's
            `pickle` module will be used. This places quite stringent restrictions on
            the object types used in the computation. In particular,

            - non-deterministic values retrieved in the function
            - return value
            - function itself

            must be picklable. This excludes lambda functions and bound methods, among
            other things. For a full list, please consult [3]_.

    Notes
    -----
    This function is essentially an implementation of McCarty's ``amb`` operator [1]_.

    The goal of this function is to give an illusion of the computation forking at each
    decision point. Nevertheless, due to Python's limitations, each execution path is in
    fact a separate call to the supplied computation - from start to end. This has some
    consequences on the design and intended usage of this function.

    - The list of choices in a given node of the decision tree is determined by the
      first execution path that includes it. Choices passed to the `Amb` object at that
      node during subsequent executions are ignored.

    - Expensive computations should be performed once and have their results shared
      between execution paths with a common prefix, using the `once` or the `Thunk`
      class.

    References
    ----------
    .. [1] J. McCarthy, "A basis for a mathematical theory of computation", Computer
           Programming and Formal Systems, North-Holland, Amsterdam, 1963, pp. 33–70.
    .. [2] https://docs.python.org/3/library/concurrent.futures.html
    .. [3] https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

    Examples
    --------
    Compute a list of Pythagorean triplets.

    >>> def triplets(amb):
    ...     a = amb("a", range(1, 6))
    ...     b = amb("b", range(1, 6))
    ...     c = amb("c", range(1, 6))
    ...     amb.ensure(a**2 + b**2 == c**2)
    ...     return (a, b, c)
    >>> [a.result for a in nondeterministic(triplets)]
    [(3, 4, 5), (4, 3, 5)]

    Compute a list of subsets.

    >>> def subsets(xs):
    ...     def go(amb):
    ...         return [x for x in xs if amb("pick", [False, True])]
    ...     return [s.result for s in nondeterministic(go)]
    >>> subsets([1, 2, 3])
    [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]

    Iterate through a complex parameter set in parallel.

    >>> def experiment(amb):
    ...     method = amb("method", ["abc", "evolution"])
    ...     if method == "abc":
    ...         eps = amb("eps", [0.1, 0.4])
    ...         return method, eps
    ...     else:
    ...         alg = amb("algorithm", ["DE", "PSO"])
    ...         return method, alg
    >>> paths = nondeterministic(experiment, execution="threads")
    >>> sorted([p.result for p in paths])
    [('abc', 0.1), ('abc', 0.4), ('evolution', 'DE'), ('evolution', 'PSO')]
    """
    environment = _choose_environment(execution, **kwargs)

    runner = _NondeterministicExecutor(
        environment,
        event_filter=_EventFilter(node_events, aborted),
        pass_exceptions=pass_exceptions,
    )

    match mode:
        case "sync" | Mode.SYNC:
            return runner.run(fun)
        case "async" | Mode.ASYNC:
            return runner.async_run(fun)
        case _:
            raise ValueError(f"Unknown mode '{mode}'")
