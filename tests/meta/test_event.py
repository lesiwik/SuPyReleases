from typing import Annotated

import supy.meta as meta
from supy.meta.annotations import Param, event
from supy.utils import Range

# There is a lot of type-annotations here inside untyped functions,
# but these annotations are for the purpose of testing extraction
# code only.
# mypy: disable-error-code = annotation-unchecked


def test_extracts_empty_event():
    @event(label="Empty event")
    class EmptyEvent:
        pass

    (info,) = meta.events.all()
    assert info == meta.EventType(
        name="EmptyEvent",
        label="Empty event",
        desc=None,
        type=EmptyEvent,
        params=(
            meta.Param(
                name="time",
                label=None,
                desc="Time of the event",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
        ),
    )


def test_extracts_event():
    @event(label="Some fancy event", desc="Things are happening")
    class SomeEvent:
        amount: Annotated[
            float, Param("amount of stuff", desc="The quantity of stuff in kg")
        ]
        foo: Annotated[bool, Param("Foobility", desc="Should it foo or not")] = False

    (info,) = meta.events.all()
    assert info == meta.EventType(
        name="SomeEvent",
        label="Some fancy event",
        desc="Things are happening",
        type=SomeEvent,
        params=(
            meta.Param(
                name="time",
                label=None,
                desc="Time of the event",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
            meta.Param(
                name="amount",
                label="amount of stuff",
                desc="The quantity of stuff in kg",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
            meta.Param(
                name="foo",
                label="Foobility",
                desc="Should it foo or not",
                type=bool,
                required=False,
                range=Range.UNKNOWN,
            ),
        ),
    )
