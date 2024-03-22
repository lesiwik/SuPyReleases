from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, NewType, TypeAlias, TypeVar

from supy.utils import Range

_T_co = TypeVar("_T_co", covariant=True)


@dataclass(frozen=True)
class Entity:
    """
    Base class of named entities.

    Attributes
    ----------
    name : str
        Name of the entity.
    label : str, optional
        Text used to describe the entity for display purposes.
    desc : str, optional
        Short description of the entity.
    """

    name: str
    label: str | None
    desc: str | None


@dataclass(frozen=True)
class Param(Entity, Generic[_T_co]):
    """
    Description of a parameter of a model, assimilation algorithm etc.

    Attributes
    ----------
    type : type
        Type of the parameter values.
    required : bool
        If ``True``, parameter value must be specified.
    range : Range
        Range of valid/reasonable parameter values.
    """

    type: type[_T_co]
    required: bool
    range: Range[_T_co]


TimeSeries = NewType("TimeSeries", Sequence[float])
"""
Type of output items that should be interpreted as time series.
"""


@dataclass(frozen=True)
class OutputVar(Entity):
    """
    Description of a single output quantity.

    The most common output of a model or a supermodel is a time series, that is, a
    sequence of numbers corresponding to some quantity at points in time specified as an
    argument to the model evaluation function. Some models/supermodels may be producing
    a sequence of numbers that is not to be interpreted as a time series. To distinguish
    these two cases, type of the output variable representing time series should be
    `TimeSeries`.

    Attributes
    ----------
    type : type
        Type of the output variable.
    range : Range
        Range of values of this output quantity.
    """

    type: type
    range: Range[float]


@dataclass(frozen=True)
class OutputGroup(Entity):
    """
    Description of a group of output items.

    Attributes
    ----------
    items : sequence of `OutputVar` and `OutputGroup`
        Output items comprising the group.
    """

    items: Sequence[OutputItem]


OutputItem: TypeAlias = OutputVar | OutputGroup
"""
Type alias for the union of all output items.
"""


@dataclass(frozen=True)
class EventType(Entity):
    """
    Description of an event type.

    Attributes
    ----------
    type : type
        Event class.
    params : tuple of `Param`
        Arguments of the event class constructor.
    """

    type: type
    params: tuple[Param, ...]


@dataclass(frozen=True)
class Model(Entity):
    """
    Description of a model.

    Attributes
    ----------
    params : tuple of `Param`
        Parameters that need to be passed to the model to evaluate it.
    events : tuple of `EventType`
        Types of events handled by the model.
    output : tuple of `OutputItem`
        Structure of output produced by the model.
    new_instance : callable
        A function which, when called with no arguments, creates a new instance of the
        model.
    """

    params: tuple[Param, ...]
    events: tuple[EventType, ...]
    output: tuple[OutputItem, ...]
    new_instance: Callable[[], object]


@dataclass(frozen=True)
class SuperModel(Entity):
    """
    Description of a supermodeling method.

    Since in general the configuration required by a supermodel depends heavily on the
    submodels it consolidates, its description is not available statically. To obtain
    it, create an instance of the supermodel using `new_instance`, and query the
    resulting supermodel instance object.

    Attributes
    ----------
    new_instance : callable
        A function which, when called with a sequence of submodels, creates a new
        instance of the supermodel.
    """

    new_instance: Callable[[Sequence[object]], object]


@dataclass(frozen=True)
class AssimilationAlgorithm(Entity):
    """
    Description of an assimilation algorithm.

    Attributes
    ----------
    params : tuple of `Param`
        Parameters that can be passed to the algorithm.
    """

    params: tuple[Param, ...]
