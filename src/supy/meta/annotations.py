from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import MISSING, dataclass
from textwrap import dedent
from typing import TypeAlias, TypeVar

import supy.meta as meta
from supy.utils import Range


@dataclass
class Param:
    """
    Additional information about a parameter.

    Use with `typing.Annotated` to provide additional information about a parameter in a
    Python type hing.

    Arguments
    ---------
    label : str, optional
        Text used to describe the parameter for display purposes.
    desc : str, optional
        Short description of the parameter.
    range : Range, optional
        Range of valid/reasonable parameter values.

    Examples
    --------
    Field of a dataclass.

    >>> @dataclass
    ... class SomeModel:
    ...     q_x: Annotated[float, Param(desc="Some model parameter")]

    Function parameter.

    >>> def fun(x: Annotated[float, Param(desc="Some param")]):
    ...     ...
    """

    label: str | None = None
    desc: str | None = None
    range: Range = Range.UNKNOWN


@dataclass
class Var:
    """
    Description of an output quantity.

    Arguments
    ---------
    name : str
        Name of the quantity.
    label : str, optional
        Text used to describe the quantity for display purposes.
    desc : str, optional
        Short description of the quantity.
    range : Range, optional
        Range of valid/reasonable parameter values.
    """

    name: str
    label: str | None = None
    desc: str | None = None
    type: type = meta.TimeSeries
    range: Range = Range.UNKNOWN


@dataclass
class Group:
    """
    Description of a group of output quantities.

    Arguments
    ---------
    name : str
        Name of the group.
    items : sequence of Group or Var
        Items comprising the group.
    label : str, optional
        Text used to describe the group for display purposes.
    desc : str, optional
        Short description of the group.
    """

    name: str
    items: Sequence[Group | Var]
    label: str | None = None
    desc: str | None = None


def _reformat_desc(desc: str) -> str:
    return dedent(desc).strip()


def _reformat_if_exists(desc: str | None) -> str | None:
    return _reformat_desc(desc) if desc is not None else None


def _parse_param(field: dataclasses.Field) -> meta.Param:
    ann = field.type
    (param,) = ann.__metadata__

    # required parameter = field with no default value
    required = (field.default, field.default_factory) == (MISSING, MISSING)

    args = typing.get_args(ann)
    field_type = args[0]

    return meta.Param(
        name=field.name,
        label=param.label,
        desc=_reformat_if_exists(param.desc),
        type=field_type,
        required=required,
        range=param.range,
    )


def _parse_var(var: Var) -> meta.OutputVar:
    return meta.OutputVar(
        name=var.name,
        label=var.label,
        desc=_reformat_if_exists(var.desc),
        type=var.type,
        range=var.range,
    )


def _parse_output_def(items: Sequence[Var | Group]) -> tuple[meta.OutputItem, ...]:
    def parse(item: Var | Group) -> meta.OutputItem:
        match item:
            case Var():
                return _parse_var(item)
            case Group() as group:
                return meta.OutputGroup(
                    name=group.name,
                    label=group.label,
                    desc=_reformat_if_exists(group.desc),
                    items=tuple(parse(v) for v in group.items),
                )

    return tuple(parse(item) for item in items)


def _get_class_var_seq(cls: type, var: str):
    try:
        value = getattr(cls, var)
    except AttributeError:
        msg = f"Class '{cls}' is not a valid model: missing '{var}'"
        raise ValueError(msg) from None
    if not isinstance(value, Iterable):
        raise TypeError(
            f"'{cls}' is not a valid model definition: "
            f"'{var}' must be a sequence class attribute"
        )
    else:
        return tuple(value)


def _vars(output: Sequence[meta.OutputItem]) -> Iterator[meta.OutputVar]:
    def gather(item: meta.OutputItem) -> Iterator[meta.OutputVar]:
        match item:
            case meta.OutputVar() as var:
                yield var
            case meta.OutputGroup() as group:
                yield from _vars(group.items)

    for item in output:
        yield from gather(item)


def _output_def_from_ode(cls: type) -> tuple[meta.OutputItem, ...]:
    state = _get_class_var_seq(cls, "state_vars")
    derived = _get_class_var_seq(cls, "derived_vars")
    valid_names = set().union(state, derived)

    output_def = _get_class_var_seq(cls, "output")
    output = _parse_output_def(output_def)

    for var in _vars(output):
        if var.name not in valid_names:
            raise ValueError(f"Unknown variable '{var.name}'")
        if var.type is not meta.TimeSeries:
            raise TypeError("ODE model output variable type must be TimeSeries")

    return output


def _params_from_dataclass(cls: type) -> tuple[meta.Param, ...]:
    return tuple([_parse_param(field) for field in dataclasses.fields(cls)])


def _events_from_dataclass(cls: type) -> tuple[meta.EventType, ...]:
    event_classes = _get_class_var_seq(cls, "events")
    return tuple(meta.events[e.__name__] for e in event_classes)


def _model_from_ode(label: str | None, desc: str | None, cls: type) -> meta.Model:
    params = _params_from_dataclass(cls)
    output = _output_def_from_ode(cls)
    events = _events_from_dataclass(cls)

    return meta.Model(
        name=cls.__name__,
        label=label,
        desc=_reformat_if_exists(desc),
        params=params,
        events=events,
        output=output,
        new_instance=None,  # TODO unimplemented
    )


_ModelDef: TypeAlias = type


def _model_from(label: str | None, desc: str | None, model: _ModelDef) -> meta.Model:
    return _model_from_ode(label, desc, model)


_M = TypeVar("_M", bound=_ModelDef)


def model(label: str | None = None, desc: str | None = None) -> Callable[[_M], _M]:
    def inner(model_def: _M) -> _M:
        model = _model_from(label, desc, model_def)
        meta.models.register(model)
        return model_def

    return inner


def _transform_into_event(cls: type) -> type:
    from supy.problem import Event

    as_dataclass: type = dataclasses.dataclass(frozen=True)(cls)
    new_cls = types.new_class(cls.__name__, (as_dataclass, Event))
    return dataclasses.dataclass(frozen=True)(new_cls)


def _event_from(label: str | None, desc: str | None, cls: type) -> meta.EventType:
    params = _params_from_dataclass(cls)

    return meta.EventType(
        name=cls.__name__,
        label=label,
        desc=_reformat_if_exists(desc),
        type=cls,
        params=params,
    )


def event(label: str | None = None, desc: str | None = None) -> Callable[[type], type]:
    def inner(cls: type) -> type:
        event_cls = _transform_into_event(cls)
        event_info = _event_from(label, desc, event_cls)
        meta.events.register(event_info)
        return event_cls

    return inner
