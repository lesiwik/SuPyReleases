from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from supy import Interval
from supy.meta.annotations import Param


@dataclass(frozen=True)
class Event:
    time: Annotated[float, Param(desc="Time of the event")]


@dataclass
class Problem:
    ground_truth: dict[str, NDArray[np.float_]]
    time_points: NDArray[np.float_]
    learning_window: Interval
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        if self.time_points.ndim != 1:
            raise ValueError(
                f"Time axis is not one-dimensional (shape {self.time_points.shape})"
            )
        for name, data in self.ground_truth.items():
            if data.ndim != 1:
                raise ValueError(
                    f"Value of '{name}' is not one-dimensional (shape {data.shape})"
                )
            if data.size != self.time_points.size:
                raise ValueError(
                    f"Length of '{name}' does not match the time axis "
                    f"({data.size} != {self.time_points.size})"
                )

        # ensure event list is sorted
        self.events.sort(key=lambda e: e.time)

    @property
    def domain(self) -> Interval:
        first = self.time_points[0]
        last = self.time_points[-1]
        return Interval(first, last)

    @property
    def variables(self) -> list[str]:
        return list(self.ground_truth)
