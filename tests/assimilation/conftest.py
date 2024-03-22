from typing import Any

import numpy as np
import pytest

from supy import Interval
from supy.assimilation.problem import AssimilationProblem
from supy.utils import set_seed


def _dict_norm(mapping: dict[Any, float]):
    arr = np.array(list(mapping.values()))
    return np.linalg.norm(arr)


class ExampleProblem:
    def __init__(self, model, reference_params, param_bounds):
        self.model = model
        self.reference_params = reference_params
        self.param_bounds = param_bounds

        ground_truth = model(**reference_params)
        self.problem = AssimilationProblem(model, param_bounds, ground_truth)

    def assert_solution_good_enough(self, result, epsilon):
        __tracebackhide__ = True

        params = self.reference_params.keys()
        obtained = result.params
        exact = self.reference_params

        difference = {p: exact[p] - obtained[p] for p in params}
        error = _dict_norm(difference)

        if error >= epsilon:
            msg = f"Solution not good enough, error {error} >= {epsilon}\n"
            msg += "Parameters:\n"
            msg += "\n".join(f"  {p}: {obtained[p]}, exact: {exact[p]}" for p in params)
            raise AssertionError(msg)


@pytest.fixture()
def parabola():
    def model(a, b, c):
        xs = np.linspace(-1, 1, num=10)
        return {"data": a * xs**2 + b * xs + c}

    param_bounds = {
        "a": Interval(0.8, 1.2),
        "b": Interval(1.6, 2.4),
        "c": Interval(2.4, 3.6),
    }
    reference_params = dict(a=1, b=2, c=3)
    return ExampleProblem(model, reference_params, param_bounds)


@pytest.fixture()
def sincos():
    def model(a, b):
        xs = np.linspace(-1, 1, num=20)
        return {"sin": np.sin(a * xs) + b, "cos": np.cos(a * xs) - b}

    param_bounds = {
        "a": Interval(1, 5),
        "b": Interval(1, 3),
    }
    reference_params = dict(a=4, b=1.2)
    return ExampleProblem(model, reference_params, param_bounds)


@pytest.fixture(autouse=True)
def _random_seed():
    """
    Set stable seed for `random` and `numpy.random` random number generators.

    Since most assimilation algorithms are stochastic, to ensure robust test we need to
    make sure the value of seed they use is fixed.

    This fixture is automatically requested by all the tests in ``tests/assimilation``.
    """
    set_seed(11111)
