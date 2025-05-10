"""Controller functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from dynamics import ABCDynamicsSimulator


class Controller(ABC):  # noqa: D101
    @abstractmethod
    def __init__(self,
            guidance: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
            following: Controller = None,
            custom_control_command: Optional[Callable] = None) -> None:
        """ABC initialiser.

        Parameters
        ----------
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        following: Controller
            Controller to be placed after this controller (the output of this controller
            is the input of the following controller)
        custom_control_command: Callable
            Called insted of the default control_command method.
            It must have the same signatures as the control commands.
        """
        self.guidance = guidance
        self.following = following
        self.custom_control_command = custom_control_command

    def full_control_command(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Compute the full control command given the state of the dynamics simulator.

        If there is a .following controller, the output control command will be the
        control command of following receiving as input the control command of this
        object.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        previous_control_command: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        if self.following is None:
            return self.control_command(dynamics_simulator, t, previous_control_command)
        return self.following.full_control_command(dynamics_simulator, t,
                                            self.control_command(dynamics_simulator, t,
                                                                previous_control_command))

    def control_command(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Compute the control command given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        previous_control_command: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        if self.custom_control_command is not None:
            return self.custom_control_command(dynamics_simulator, t,
                                                previous_control_command)
        return self._default_control_command(dynamics_simulator, t,
                                                previous_control_command)

    @abstractmethod
    def _default_control_command(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default control command given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        previous_control_command: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        return


class PDController(Controller):
    """PD controller.

    Attributes
    ----------
    kp: float | ndarray
        Proportional gains
    kd: float | ndarray
        Derivative gains
    guidance: callable
        Guidance law to be applied to calculate the state difference.
        The guidance law should take the time as first input and the state as second
    following: Controller
        Controller to be placed after this controller (the output of this controller
        is the input of the following controller)
    """

    def __init__(self, kp: Union[float, np.ndarray], kd: Union[float, np.ndarray],
                    guidance: Callable[[float, np.ndarray], np.ndarray],
                    following: Controller = None,
                    custom_control_command: Optional[Callable] = None) -> None:
        """Initialise the PD controller.

        Parameters
        ----------
        kp: float | ndarray
            Proportional gains
        kd: float | ndarray
            Derivative gains
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        following: Controller
            Controller to be placed after this controller (the output of this controller
            is the input of the following controller)
        custom_control_command: Callable
            Called insted of the default control_command method.
            It must have the same signatures as the control commands.
        """
        super().__init__(guidance, following, custom_control_command)
        self.kp = kp
        self.kd = kd

    def _default_control_command(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default control command given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        previous_control_command: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        sc = dynamics_simulator.spacecraft
        ref = self.guidance(t, sc.attitude.x)
        e, e_dot = sc.attitude.state_error(ref[:-3], ref[-3:], sc.mean_motion)
        if len(e) == 4:
            e_dot = np.append(e_dot, 0)

        # with quaternions it only takes the first 3 components
        return self.control_law(e, e_dot)

    def control_law(self, e: Union[float, np.ndarray],
                e_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the control variable given input error and its derivative.

        Parameters
        ----------
        e: float | ndarray
            Input error
        e_dot: float | ndarray
            Input error derivative

        Returns
        -------
        float | ndarray
            Control command.
        """
        return -(e * self.kp + e_dot * self.kd)


class NDIModelBased(Controller):
    """Nonlinear Dynamic Inversion - model based approach.

    Requires a previous controller, since it takes its control command as input

    Attributes
    ----------
    following: Controller
        Controller to be placed after this controller (the output of this controller
        is the input of the following controller)
    """

    def __init__(self, following: Controller = None,
                    custom_control_command: Optional[Callable] = None) -> None:
        """Initialise the PD controller.

        Parameters
        ----------
        following: Controller
            Controller to be placed after this controller (the output of this controller
            is the input of the following controller)
        custom_control_command: Callable
            Called insted of the default control_command method.
            It must have the same signatures as the control commands.
        """
        super().__init__(None, following, custom_control_command)

    def _default_control_command(self,  # noqa: PLR6301
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: np.ndarray,
                        ) -> np.ndarray:
        """Set default control command given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time.
        previous_control_command: ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        ds = dynamics_simulator
        return ds.inv_gmatrix() @ (previous_control_command[:3] - ds.fx())


class NDITimeScaleSeparation(NDIModelBased):
    """Nonlinear Dynamic Inversion - time-scale separation approach.

    Requires a previous controller, since it takes its control command as input

    Attributes
    ----------
    following: Controller
        Controller to be placed after this controller (the output of this controller
        is the input of the following controller)
    """

    def __init__(self, following: Controller = None,
                    custom_control_command: Optional[Callable] = None) -> None:
        """Initialise the PD controller.

        Parameters
        ----------
        following: Controller
            Controller to be placed after this controller (the output of this controller
            is the input of the following controller)
        custom_control_command: Callable
            Called insted of the default control_command method.
            It must have the same signatures as the control commands.
        """
        super().__init__(following, custom_control_command)

    def _default_control_command(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        previous_control_command: np.ndarray,
                        ) -> np.ndarray:
        """Set default control command given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time.
        previous_control_command: ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        ds = dynamics_simulator
        outer_loop_control = np.linalg.inv(ds.spacecraft.attitude.w2angdot_matrix()) \
                                    @ previous_control_command
        return super()._default_control_command(ds, t, outer_loop_control)
