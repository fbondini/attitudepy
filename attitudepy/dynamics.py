"""Attitude dynamics functions."""
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import odeint

from .controller import Controller
from .spacecraft_class import Spacecraft


class ABCDynamicsSimulator(ABC):
    """Abstract class for defining dynamical equations and integration."""

    def __init__(self, spacecraft: Spacecraft, control: Controller = None) -> None:
        """Initialise the dynamics simulator.

        Parameters
        ----------
        spacecraft: Spacecraft
            Spacecraft object
        control: Controller (optional)
            Controller object
        """
        self.spacecraft = spacecraft
        self.control = control

    def simulate(self, tvect: np.ndarray) -> np.ndarray:
        """Integrates the dynamical equations at the provided timesteps.

        Parameters
        ----------
        tvect: ndarray
            Vector of times at which the dynamics must be integrated.

        Returns
        -------
        ndarray
            y output of scipy.integrate.odeint
        """
        return odeint(self.dynamics_equation, self.spacecraft.attitude.x,
                            tvect, (self,))

    @staticmethod
    @abstractmethod
    def dynamics_equation(x: np.ndarray, t: float,
                            dynamics_simulator: "ABCDynamicsSimulator") -> np.ndarray:
        """Define the dynamics differential equations without gravity torque.

        To be passed to the integrator to integrate the state.

        Parameters
        ----------
        t: float
            Time (s).
        x: np.ndarray
            Current attitude state (either 6 components for eul angles + w,
            or 7 for quats + w).
        sc: Spacecraft
            Spacecraft object.
        dynamics_simulator: ABCDynamicsSimulator
            Dynamics simulator object.

        Returns
        -------
        ndarray
            Attitude state derivative.
        """
        return

    @abstractmethod
    def fx(self) -> np.ndarray:
        """f(x) function.

        It represents the part of the dynamics equations that do not depend on the
        external moments.

        Returns
        -------
        ndarray
            Evaluation of f(x) at the current state.
        """
        return

    @abstractmethod
    def gmatrix(self) -> np.ndarray:
        """G(x) matrix.

        It represents the part of the dynamics equations that depend linearly on
        the external moments u (or the lienarisation of the G(x,u) function if it
        is not linearly dependent on u).

        Returns
        -------
        ndarray
            Evaluation of G(x) at the current state.
        """
        return

    @abstractmethod
    def inv_gmatrix(self) -> np.ndarray:
        """G(x)^-1 matrix.

        It represents inverse of the part of the dynamics equations that depend linearly
        on the external moments u (or the lienarisation of the G(x,u) function if it
        is not linearly dependent on u).

        Returns
        -------
        ndarray
            Evaluation of G(x) at the current state.
        """
        return


class DynamicsSimulatorNoGravityTorque(ABCDynamicsSimulator):
    """Class defining dynamical equations and integration with no gravity torque."""

    def __init__(self, spacecraft: Spacecraft, control: Controller = None) -> None:
        """Initialise the dynamics simulator.

        Parameters
        ----------
        spacecraft: Spacecraft
            Spacecraft object
        control: Controller (optional)
            Controller object
        """
        super().__init__(spacecraft, control)

    @staticmethod
    def dynamics_equation(x: np.ndarray, t: float,
                    dynamics_simulator: ABCDynamicsSimulator) -> np.ndarray:
        """Define the dynamics differential equations without gravity torque.

        To be passed to the integrator to integrate the state.

        Parameters
        ----------
        t: float
            Time (s).
        x: np.ndarray
            Current attitude state (either 6 components for eul angles + w,
            or 7 for quats + w).
        dynamics_simulator: ABCDynamicsSimulator

        Returns
        -------
        xdot: float
            Attitude state derivative.
        """
        sc = dynamics_simulator.spacecraft
        ctrl = dynamics_simulator.control

        sc.attitude.ang = x[0:-3]
        sc.attitude.w = x[-3:]

        if len(sc.attitude.ang) == 4:
            sc.attitude.ang = sc.attitude.ang / np.linalg.norm(sc.attitude.ang)

        angdot = sc.attitude.kinematic_diff_equation(sc.mean_motion)

        u = ctrl.full_control_command(dynamics_simulator, t)[:3] + sc.torque_disturb if ctrl is not None else np.zeros(3)  # noqa: E501

        return np.append(angdot, dynamics_simulator.fx() + dynamics_simulator.gmatrix() @ u)  # noqa: E501

    def fx(self) -> np.ndarray:
        """f(x) function.

        It represents the part of the dynamics equations that do not depend on the
        external moments.

        Returns
        -------
        ndarray
            Evaluation of f(x) at the current state.
        """
        sc = self.spacecraft
        return -np.linalg.inv(sc.inertia) @ np.cross(sc.attitude.w,
                                                        sc.inertia @ sc.attitude.w)

    def gmatrix(self) -> np.ndarray:
        """G(x) matrix.

        It represents the part of the dynamics equations that depend linearly on
        the external moments u (or the lienarisation of the G(x,u) function if it
        is not linearly dependent on u).

        Returns
        -------
        ndarray
            Evaluation of G(x) at the current state.
        """
        return np.linalg.inv(self.spacecraft.inertia)

    def inv_gmatrix(self) -> np.ndarray:
        """G(x)^-1 matrix.

        It represents inverse of the part of the dynamics equations that depend linearly
        on the external moments u (or the lienarisation of the G(x,u) function if it
        is not linearly dependent on u).

        Returns
        -------
        ndarray
            Evaluation of G(x) at the current state.
        """
        return self.spacecraft.inertia


class DynamicsSimulator(DynamicsSimulatorNoGravityTorque):
    """Class defining the standard dynamical equations and integration."""

    def __init__(self, spacecraft: Spacecraft, control: Controller = None) -> None:
        """Initialise the dynamics simulator.

        Parameters
        ----------
        spacecraft: Spacecraft
            Spacecraft object
        control: Controller (optional)
            Controller object
        """
        super().__init__(spacecraft, control)

    @staticmethod
    def dynamics_equation(x: np.ndarray, t: float,
                                dynamics_simulator: ABCDynamicsSimulator) -> np.ndarray:
        """Define the dynamics differential equations.

        To be passed to the integrator to integrate the state.

        Parameters
        ----------
        t: float
            Time (s).
        x: np.ndarray
            Current attitude state (either 6 components for eul angles + w,
            or 7 for quats + w).
        dynamics_simulator: ABCDynamicsSimulator
            Dynamics simulator object

        Returns
        -------
        xdot: float
            Attitude state derivative.
        """
        sc = dynamics_simulator.spacecraft
        xdot = DynamicsSimulatorNoGravityTorque.dynamics_equation(x, t, dynamics_simulator)  # noqa: E501
        xdot[-3:] = xdot[-3:] + (np.linalg.inv(sc.inertia) @
                sc.attitude.gravity_gradient_torque(sc.mean_motion, sc.inertia))

        return xdot
