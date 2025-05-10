"""Integrator class."""
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
from scipy.integrate import odeint

PRECISION = 8  # precision digits in time steps indices


class IntegratorSettings(ABC):
    """Defines the integrator settings and sets the ODE solver."""

    def __init__(self, tspan: List) -> None:
        """Initialise the generic integrator.

        Parameters
        ----------
        tspan: list
            Start and end time of the simulation.
        """
        self.tspan = tspan

    @abstractmethod
    def integrate(self, fun: Callable, x0: np.ndarray, args: Tuple) -> dict:
        """Run the integration.

        Parameters
        ----------
        fun: Callable
            Function describing the derivative of the state.
        x0: np.ndarray
            Initial state
        args: Tuple
            Additional arguments to be passed to fun

        Returns
        -------
        state_history: dict
            Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.
        """
        return


class ScipyIntegrator(IntegratorSettings):
    """Defines the class for odeint settings and sets the ODE solver."""

    def __init__(self, t_vec: np.ndarray) -> None:
        """Initialise the Runge Kutta 4 integrator.

        Parameters
        ----------
        t_vec: ndarray
            Time vector at which to evaluate the state
        """
        self.t_vec = t_vec

    def integrate(self, fun: Callable, x0: np.ndarray, args: Tuple) -> dict:
        """Run the integration.

        Parameters
        ----------
        fun: Callable
            Function describing the derivative of the state.
        x0: np.ndarray
            Initial state
        args: Tuple
            Additional arguments to be passed to fun

        Returns
        -------
        state_history: dict
            Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.
        """
        y = odeint(fun, x0, self.t_vec, args)

        state_history = dict()
        for i, t in enumerate(self.t_vec):
            state_history[round(float(t), PRECISION)] = y[i, :]
        return state_history
