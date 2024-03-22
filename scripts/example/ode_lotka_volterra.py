# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% jupyter={"source_hidden": true}
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np

from supy.ode import ODE, solve

# %% [markdown]
# Lotka-Volterra predator-prey model ([wikipedia][1])
#
# [1]: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations


# %%
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


# %%
ts = np.linspace(0, 140, num=300)
system = LotkaVolterra(0.1, 0.02, 0.02, 0.4)

solution = solve(system, init=(10, 10), time_points=ts)
x = solution["x"]
y = solution["y"]
total = solution["total"]

# %%
plt.plot(ts, x, label="population")
plt.plot(ts, y, label="predators")
plt.legend(loc="upper right")
plt.show()
