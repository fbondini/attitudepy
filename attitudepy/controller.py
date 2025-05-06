"""Controller functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    import numpy as np


class Controller(ABC):  # noqa: D101
    @abstractmethod
    def __init__(self, guidance: Callable[[float, np.ndarray], np.ndarray]) -> None:
        """ABC initialiser.

        Parameters
        ----------
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        """
        self.guidance = guidance

    @abstractmethod
    def u(self, e: Union[float, np.ndarray],
                e_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the control variable diven input error and its derivative.

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
        return


class PDController(Controller):
    """PD controller.

    Attributes
    ----------
    kp: float | ndarray
        Proportional gains
    kd: float | ndarray
        Derivative gains
    """

    def __init__(self, kp: Union[float, np.ndarray], kd: Union[float, np.ndarray],
                        guidance: Callable[[float, np.ndarray], np.ndarray]) -> None:
        """Initialise the PD controller.

        Attributes
        ----------
        kp: float | ndarray
            Proportional gains
        kd: float | ndarray
            Derivative gains
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        """
        super().__init__(guidance)
        self.kp = kp
        self.kd = kd

    def u(self, e: Union[float, np.ndarray],
                e_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the control variable diven input error and its derivative.

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
