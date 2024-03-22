from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

from supy.interval import Interval


@dataclass(frozen=True)
class AssimilationProblem:
    model: Callable[..., dict[str, NDArray[np.float_]]]
    param_bounds: dict[str, Interval]
    ground_truth: dict[str, NDArray[np.float_]]

    @property
    def params(self):
        return self.param_bounds.keys()


@dataclass(frozen=True)
class AssimilationResult:
    params: dict[str, float]
    model_calls: int
    total_time: timedelta
    model_time: timedelta
