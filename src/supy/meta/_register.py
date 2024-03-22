from threading import Lock
from typing import Generic, TypeVar

from supy.meta._data import AssimilationAlgorithm, Entity, EventType, Model, SuperModel

_T = TypeVar("_T", bound=Entity)


class Register(Generic[_T]):
    """
    Register of entities, allowing listing and lookup by name.

    All the supported operations are thread-safe.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _T] = {}
        self._lock = Lock()

    def register(self, item: _T) -> None:
        """
        Add new item to the register.

        Name is extracted from the `name` attribute of the registered item.

        Parameters
        ----------
        item : T
            New item to register.
        """
        with self._lock:
            self._entries[item.name] = item

    def all(self) -> tuple[_T, ...]:
        """
        Return a tuple of all registered items.

        Returns
        -------
        tuple of T
            Tuple of all registered items.
        """
        with self._lock:
            return tuple(self._entries.values())

    def __getitem__(self, name: str) -> _T:
        """
        Look up an item by name.

        Parameters
        ----------
        name : str
            Name of the desired item.

        Returns
        -------
        T
            Item with specified name, if available.

        Raises
        ------
        KeyError
            If no item with specified name has been registered.
        """
        with self._lock:
            return self._entries[name]

    def clear(self) -> None:
        """Remove all registered items."""
        with self._lock:
            self._entries = {}


models = Register[Model]()
"""Register of available models."""

events = Register[EventType]()
"""Register of available event classes."""

supermodels = Register[SuperModel]()
"""Register of available supermodeling methods."""

assimilation_algorithms = Register[AssimilationAlgorithm]()
"""Register of available assimilation algorithms."""
