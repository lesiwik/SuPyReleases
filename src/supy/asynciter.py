"""Building blocks for data structures supporting asynchronous iteration."""
from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Generic, Self, TypeVar, overload

from supy.utils import spawn_task

# used by AsyncPushIterator to denote end of iteration
_sentinel = object()


T = TypeVar("T")


class AsyncPushIterator(Generic[T]):
    """
    Asynchronous iterator that can be fed items one at a time.

    Notes
    -----
    The items added to the iterator are internally stored in a queue before being
    returned in `__anext__`. To avoid ``O(N)`` memory usage, ensure prompt iterator
    consumption.
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self._closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        while True:
            item = await self._queue.get()
            if item is _sentinel:
                raise StopAsyncIteration()
            else:
                return item

    def add(self, item: T) -> None:
        """
        Notify the iterator about a new item.

        Parameters
        ----------
        item : T
            New item to pass to the iterator consumer.

        Raises
        ------
        ValueError
            If the iterator has already been closed.
        """
        if self._closed:
            raise ValueError("AsyncPushIterator is closed")

        self._queue.put_nowait(item)

    def close(self) -> None:
        """
        Notify the iterator about the end of item stream.

        Raises
        ------
        ValueError
            If the iterator has already been closed.
        """
        self.add(_sentinel)  # type: ignore
        self._closed = True


class _AsyncBufferIterator(Generic[T]):
    # Unlike AsyncPushIterator, there is no internal queue.
    # Iterates directly over the contents of the associated AsyncBuffer.
    def __init__(self, buff: AsyncBuffer[T]):
        self._idx = -1
        self._buffer = buff

    def __aiter__(self) -> Self:
        return self

    def _has_next(self) -> bool:
        return self._idx < len(self._buffer)

    async def __anext__(self) -> T:
        self._idx += 1
        buff = self._buffer

        async with buff._condvar:
            await buff._condvar.wait_for(lambda: buff._closed or self._has_next())

            # Buffer may already be closed here, but if we still have items left,
            # we continue the iteration.
            if self._has_next():
                return buff._items[self._idx]
            else:
                raise StopAsyncIteration()


class AsyncBuffer(Generic[T]):
    """
    Sequence supporting asynchronous iteration while adding new items.

    Asynchronous iteration produces all the items currently in the buffer, and waits for
    new items to appear. End of the sequence is signaled by calling the `close` method.

    It can be used as a context manager, in which case the buffer is automatically
    closed upon exit.

    Notes
    -----
    Since asynchronous iterators wait for new items upon reaching the end of the buffer,
    if `close` is never called, the iteration process never terminates.

    Examples
    --------
    Adding items after initiating iteration.

    >>> async def simple():
    ...     buff = AsyncBuffer()
    ...     async def collect():
    ...         async for x in buff:
    ...             print(x)
    ...     items = asyncio.create_task(collect())
    ...     buff.add(1)
    ...     buff.add(2)
    ...     buff.close()
    >>> asyncio.run(simple())
    1
    2

    As a context manager.

    >>> async def gather_items():
    ...     with AsyncBuffer() as buff:
    ...         buff.add(1)
    ...         buff.add(2)
    ...     return [x async for x in buff]
    >>> asyncio.run(gather_items())
    [1, 2]
    """

    def __init__(self) -> None:
        self._items: list[T] = []
        self._closed = False
        self._condvar = asyncio.Condition()

    # Since the conditional variable does not have any inherent meaning outside
    # the context of the currently executing program, we discard it during pickling
    # and recreate it at load.

    def __getstate__(self) -> tuple:
        return (self._items, self._closed)

    def __setstate__(self, state: tuple) -> None:
        self._items, self._closed = state
        self._condvar = asyncio.Condition()

    def add(self, item: T) -> None:
        """
        Add new item to the buffer.

        Parameters
        ----------
        item : T
            New item.

        Raises
        ------
        ValueError
            If the buffer has already been closed.
        """
        self._ensure_open()
        self._items.append(item)
        self._notify_iters()

    def _notify_iters(self) -> None:
        # Awaken all the iterators
        async def do_notify():
            async with self._condvar:
                self._condvar.notify_all()

        spawn_task(do_notify(), "Notify AsyncBuffer iterators about new item")

    def close(self) -> None:
        """
        Close the buffer.

        No new items may be added after calling this method.

        Raises
        ------
        ValueError
            If the buffer has already been closed.
        """
        self._ensure_open()
        self._closed = True
        self._notify_iters()

    @staticmethod
    def from_async_iterable(items: AsyncIterable[T]) -> AsyncBuffer[T]:
        """
        Asynchronously fill buffer with values from async iterable.

        This method returns a new buffer immediately, scheduling filling it with values
        from passed iterable in the background. Buffer is automatically closed once the
        elements in the source iterable are exhausted.

        Parameters
        ----------
        items : async iterable of T
            Asynchronous iterable or generator that supply items.

        Returns
        -------
        AsyncBuffer
            Buffer with items from the supplied iterable.
        """
        buff = AsyncBuffer[T]()

        async def go():
            with buff:
                async for item in items:
                    buff.add(item)

        spawn_task(go(), "Building AsyncBuffer from iterable")
        return buff

    def __aiter__(self) -> AsyncIterator[T]:
        return _AsyncBufferIterator(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("AsyncBuffer is closed")

    def __len__(self) -> int:
        return len(self._items)


K = TypeVar("K")
V = TypeVar("V")


class AsyncDict(Generic[K, V]):
    """
    Mutable mapping allowing to ``await`` for a key to appear.

    This class offers an interface similar to `collections.abc.MutableMapping`, with a
    few notable differences:

    - `__getitem__` and `pop` are coroutines, and thus return an `Awaitable[V]` instead
      of `V`
    - `pop` does not have a default value argument, since it waits until the key appears
      in the dictionary

    Parameters
    ----------
    items : {mapping, iterable}, optional
        Mapping entries, specified as a mapping, or an iterable of ``(key, value)``
        pairs.
    **kwargs : dict, optional
        Keyword arguments specifying additional mapping entries.
    """

    def __init__(self, items=(), /, **kwargs) -> None:
        self._dict = dict(items, **kwargs)
        self._pending: dict[K, asyncio.Event] = {}

    # During pickling we discard the pending task list

    def __getstate__(self) -> tuple:
        return (self._dict,)

    def __setstate__(self, state: tuple) -> None:
        self._dict = state[0]
        self._pending = {}

    async def __getitem__(self, key: K) -> V:
        """
        Return the mapping value, waiting until it is available if needed.

        Parameters
        ----------
        key : K
            Mapping key.

        Returns
        -------
        awaitable
            Coroutine that produces value of type `V`.
        """
        try:
            return self._dict[key]
        except KeyError:
            event = self._get_pending_event(key)
            await event.wait()

        return self._dict[key]

    @overload
    def get(self, key: K) -> V | None:
        ...

    @overload
    def get(self, key: K, default: T) -> V | T:
        ...

    def get(self, key, default=None):
        """
        Return the mapping value, or the default value if not available.

        Unlike the indexing operator, this method does not wait until the mapping value
        becomes available. Instead, if the key is not present, it returns `default`
        immediately.

        Parameters
        ----------
        key : K
            Mapping key.
        default : object, optional
            Default value to return in case the key is missing.

        Returns
        -------
        V or object
            Mapping value, if available, otherwise `default`.
        """
        try:
            return self._dict[key]
        except KeyError:
            return default

    def __setitem__(self, key: K, value: V) -> None:
        """
        Set the mapping value.

        All the tasks that have been asynchronously waiting for it are awakened.

        Parameters
        ----------
        key : K
            Mapping key.
        value : V
            Mapping value.
        """
        self._dict[key] = value

        if key in self._pending:
            self._awake_pending(key)

    def __delitem__(self, key: K) -> None:
        """
        Remove the entry from the mapping.

        Parameters
        ----------
        key : K
            Mapping key.
        """
        del self._dict[key]

    async def pop(self, key: K) -> V:
        """
        Remove and return the entry from the mapping, waiting for it if needed.

        Parameters
        ----------
        key : K
            Mapping key.

        Returns
        -------
        awaitable
            Coroutine that produces value of type `V`.
        """
        # Need to await before deletion, otherwise the __getitem__
        # coroutine will be executed after the key is deleted and
        # the key will end up in _pending.
        val = await self[key]
        del self[key]
        return val

    def popitem(self) -> tuple[K, V]:
        """
        Remove and return a ``(key, value)`` pair from the mapping.

        Returns
        -------
        (K, V)
            A ``(key, value)`` tuple.

        Raises
        ------
        KeyError
            If the mapping is empty.
        """
        return self._dict.popitem()

    def update(self, other=(), /, **kwargs) -> None:
        """
        Update the mapping with new entries, overwriting existing keys.

        As `other`, it accepts either a mapping, or an iterable of ``(key, value)``
        pairs. If keyword arguments are specified, the mapping is then updated with
        these ``(key, value)`` pairs.

        All the tasks that have been asynchronously waiting for one of the new mapping
        entries it are awakened.

        Parameters
        ----------
        other : {mapping, iterable}, optional
            Mapping entries, specified as a mapping, or an iterable of ``(key, value)``
            pairs.
        **kwargs : dict, optional
            Keyword arguments specifying additional mapping entries.
        """
        new = dict(other, **kwargs)
        self._dict.update(new)

        for key in new.keys() & self._pending:
            self._awake_pending(key)

    def __ior__(self, other) -> Self:
        """
        Update the mappign with new entries, overwriting existing keys.

        Equivalent to ``update(other)``.

        Parameters
        ----------
        other : {mapping, iterable}, optional
            Mapping entries, specified as a mapping, or an iterable of ``(key, value)``
            pairs.

        Returns
        -------
        Self
            Itself.
        """
        self.update(other)
        return self

    def _get_pending_event(self, key: K) -> asyncio.Event:
        try:
            return self._pending[key]
        except KeyError:
            event = asyncio.Event()
            self._pending[key] = event
            return event

    def _awake_pending(self, key: K) -> None:
        event = self._pending.pop(key)
        event.set()

    def __contains__(self, key: object) -> bool:
        """
        Check if the key is present.

        Parameters
        ----------
        key : object
            Mapping key.

        Returns
        -------
        bool
            Whether the key is present or not.
        """
        return key in self._dict

    def __iter__(self) -> Iterator[K]:
        return iter(self._dict)

    def keys(self) -> KeysView[K]:
        """
        Return a new view over the mapping keys.

        Returns
        -------
        KeysView
            View over the keys.
        """
        return KeysView(self._dict)

    def values(self) -> ValuesView[V]:
        """
        Return a new view over the mapping values.

        Returns
        -------
        ValuesView
            View over the values.
        """
        return ValuesView(self._dict)

    def items(self) -> ItemsView[K, V]:
        """
        Return a new view over the mapping ``(key, value)`` pairs.

        Returns
        -------
        ItemsView
            View over the ``(key, value)`` pairs.
        """
        return ItemsView(self._dict)

    def __len__(self) -> int:
        """
        Return the number of keys in the mapping.

        Returns
        -------
        int
            Number of keys.
        """
        return len(self._dict)

    def __eq__(self, other: object) -> bool:
        match other:
            case AsyncDict() | Mapping():
                return self._dict == other
            case _:
                return NotImplemented
