from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from supy.ode import ODE, solve


@dataclass
class LotkaVolterra(ODE):
    a: float
    b: float
    c: float
    d: float

    state_vars: ClassVar = ("x", "y")
    derived_vars: ClassVar = ("total",)

    def rhs(self, t: float, state) -> list[float]:
        x, y = state
        dx = (self.a - self.b * y) * x
        dy = (self.c * x - self.d) * y
        return [dx, dy]

    def derived(self, state) -> list[float]:
        x, y = state
        return [x + y]


def test_can_get_parameter_list_from_instance():
    system = LotkaVolterra(1, 2, 3, 4)
    assert system.params == ("a", "b", "c", "d")


def test_can_get_state_var_index():
    system = LotkaVolterra(1, 2, 3, 4)
    assert system.state_index("y") == 1


def test_can_get_param_index_from_instance():
    system = LotkaVolterra(1, 2, 3, 4)
    assert system.param_index("c") == 2


def test_can_get_parameter_dict():
    system = LotkaVolterra(1, 2, 3, 4)
    assert system.param_dict() == {"a": 1, "b": 2, "c": 3, "d": 4}


class DynamicState(ODE):
    derived_vars: ClassVar = ()

    def __init__(self, dim: int):
        self.dim = dim

    @property
    def state_vars(self):
        return tuple(f"var_{i}" for i in range(self.dim))

    def rhs(self, t, state):
        return np.zeros(self.dim)

    def derived(self, state):
        return []


def test_can_get_dynamic_state_vars():
    system = DynamicState(3)
    assert system.state_vars == ("var_0", "var_1", "var_2")


def test_can_get_dynamic_state_vars_index():
    system = DynamicState(3)
    assert system.state_index("var_1") == 1


class DynamicParams(ODE):
    derived_vars: ClassVar = ()
    state_vars: ClassVar = ("x",)

    def __init__(self, **params):
        self.__dict__.update(params)

    @property
    def params(self):
        return tuple(self.__dict__)

    def rhs(self, t, state):
        return [0]

    def derived(self, state):
        return []


def test_can_get_dynamic_params():
    system = DynamicParams(a=3, b=5, c=7)
    assert system.params == ("a", "b", "c")


def test_can_get_dynami_param_index():
    system = DynamicParams(a=3, b=5, c=7)
    assert system.param_index("b") == 1


def test_can_solve_ode():
    ts = np.linspace(0, 140, num=40)
    system = LotkaVolterra(0.1, 0.02, 0.02, 0.4)

    solution = solve(system, init=(10, 10), time_points=ts)
    x = solution["x"]
    y = solution["y"]
    total = solution["total"]

    assert x.shape == ts.shape
    assert y.shape == ts.shape
    assert total.shape == ts.shape

    np.testing.assert_array_less(5, x)
    np.testing.assert_array_less(x, 40)

    np.testing.assert_array_less(y, 17)
    np.testing.assert_array_less(0, y)

    np.testing.assert_allclose(total, x + y)
