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
# First, we need to define the ordinary differential equations system
# we wish to solve by specifying
#
#  - parameters
#  - state variables, which appear in the equation
#  - output variables, which are computed from the state
#  - right hand side, i.e. the time derivative of state
#
# Parameters are specified as members of the class representing the ODE.
# For that to work, the class needs to be a [dataclass][1].
# In more complex cases, when their number and names are not known in advance,
# they can be defined by overriding the `params` property.
#
# If there is only one state or output variable, remember to correctly
# specify the 1-tuple as ``("var",)``, not ``("var")``!
#
# If there are no output variables, we can use an empty tuple ``()``.
#
# [1]: https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass


# %%
@dataclass
class Decay(ODE):
    a: float

    state_vars: ClassVar = ("y",)
    derived_vars: ClassVar = ()

    def rhs(self, t: float, state) -> list[float]:
        y = state
        dy = -self.a * y
        return [dy]

    def derived(self, state) -> list[float]:
        return []


# %% [markdown]
# Instantiate and solve the system. In addition to the parameter values, we need to
# supply the initial state vector, and the time points at which the solution is to be
# computed. These time points are independent of the underlying time discretization
# scheme, so they can be chosen freely.

# %%
system = Decay(0.5)

ts = np.linspace(0, 10, num=100)
solution = solve(system, init=(1,), time_points=ts)
y = solution["y"]

# %% [markdown]
# Finally, we can plot the solution using `matplotlib`.

# %%
plt.plot(ts, y)
plt.show()
