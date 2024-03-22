import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from supy import Interval


@dataclass
class ODE(ABC):
    """
    Base class for ODE systems.

    It represents a system of ordinary differential equations, that can additionally
    modify its state in a discontinuous way in response to external events. This allows
    it to model factors that cannot be captured directly by the continuous ODE, like
    therapy in tumor simulations.
    """

    @property
    @abstractmethod
    def state_vars(self) -> tuple[str, ...]:
        """
        Return names of state variables of the system.

        These are the quantities whose time derivatives appear on the left-hand side of
        the equation.
        """

    @property
    @abstractmethod
    def derived_vars(self) -> tuple[str, ...]:
        """
        Return names of derived variables.

        These quantities are computed from the state, but do not appear directly in the
        system equations. E.g. total tumor volume computed as a sum of different types
        of cells.
        """

    @abstractmethod
    def rhs(self, t: float, state: NDArray) -> ArrayLike:
        """
        Compute the right-hand side of the equation.

        Parameters
        ----------
        t : float
            Time point.
        state : ndarray
            Vector of state variables.

        Returns
        -------
        array_like
            Vector of time derivatives of state variables at the specified time point.
        """

    @abstractmethod
    def derived(self, state: NDArray) -> ArrayLike:
        """
        Compute the derived variables form state.

        Parameters
        ----------
        state : ndarray
            Vector of state variables.

        Returns
        -------
        array_like
            Vector of output variables computed from the system state.
        """

    def handle_event(self, event, state: NDArray) -> None:  # noqa: B027
        """
        Modify the state of the system in response to an event.

        Parameters
        ----------
        event : object
            Event data.
        state : ndarray
            Vector of state variables.
        """

    def state_index(self, name: str) -> int:
        """
        Return index of the state variable.

        Parameters
        ----------
        name : str
            Name of the state variable.

        Returns
        -------
        int
            Index of the specified name in the state variable list.
        """
        return self.state_vars.index(name)

    @property
    def params(self) -> tuple[str, ...]:
        """
        Return names of ODE parameters.

        The solution of ODE should be uniquely determined by these parameters and the
        initial state vector.
        """
        return tuple(f.name for f in fields(type(self)))

    def param_index(self, name: str) -> int:
        """
        Return index of the parameter.

        Parameters
        ----------
        name : str
            Name of the parameter.

        Returns
        -------
        int
            Index of the specified name in the list of parameters.
        """
        return self.params.index(name)

    def param_dict(self) -> dict[str, Any]:
        """
        Return dictionary with parameter values of the ODE instance.

        Returns
        -------
        dict
            Dictionary of ODE parameters.
        """
        return {name: getattr(self, name) for name in self.params}


def solve(
    ode: ODE, init: ArrayLike, time_points: NDArray, **solver_options
) -> dict[str, NDArray]:
    """
    Solve an ODE system.

    Given an ODE system and initial state vector, it computes values of state and output
    variables at specified points in time.

    Parameters
    ----------
    ode : ODE
        Object representing the differential equation to solve.
    init : array_like, shape (N,)
        Initial state.
    time_points : ndarray, shape (M,)
        Times at which to store the computed solution, must be sorted.

    Returns
    -------
    dict of ndarray, shape (M,)
        Dictionary containing time series of state and output variables.

    Other Parameters
    ----------------
    **solver_options : dict, optional
        Additional arguments passed to `scipy.integrate.solve_ivp` method.

    Warns
    -----
    UserWarning
        If the ODE solver does not converge.
    """
    domain = Interval(time_points[0], time_points[-1])
    result = solve_ivp(ode.rhs, domain, init, t_eval=time_points, **solver_options)

    if not result.success:
        warnings.warn("ODE solver did not converge", stacklevel=2)

    state = result.y
    derived = np.apply_along_axis(ode.derived, axis=0, arr=state)

    state_dict = _separate_vars(state, ode.state_vars)
    derived_dict = _separate_vars(derived, ode.derived_vars)

    return state_dict | derived_dict


def _separate_vars(array: NDArray, names: Iterable[str]) -> dict[str, NDArray]:
    return {name: array[i, ...] for i, name in enumerate(names)}
