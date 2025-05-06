"""Attitude class."""

from abc import ABC, abstractmethod

import numpy as np


class Attitude(ABC):
    """Abstract base class for spacecraft attitude representations.

    Attributes
    ----------
    ang : ndarray
        Attitude representation (e.g., Euler angles or quaternion).
    w : ndarray
        Angular velocity vector (rad/s).
    """

    @abstractmethod
    def __init__(self) -> None:
        self.ang = None
        self.w = None

    @abstractmethod
    def kinematic_diff_equation(self, ang: np.ndarray) -> np.ndarray:
        """Compute the kinematic differential equation for attitude.

        Parameters
        ----------
        ang : ndarray
            Attitude vector (e.g., Euler angles or quaternion).

        Returns
        -------
        ndarray
            Time derivative of the attitude vector.
        """
        return


class AttitudeEuler(Attitude):
    """Attitude representation using Euler angles.

    Attributes
    ----------
    ang : ndarray
        Euler angles in radians.
    w : ndarray
        Angular velocity vector (rad/s).
    """

    def __init__(self, initial_eul_angles: np.ndarray):
        """Initialise attitude.

        Parameters
        ----------
        initial_eul_angles: np.ndarray
            Initial Euler angles in radians

        """
        self.ang = initial_eul_angles
        self.w = None

    def kinematic_diff_equation(self, eul: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Define the kinematic differential equation.

        Gets the euler angles derivative, used in the dynamics to be integrated.

        Parameters
        ----------
        initial_eul_angles: np.ndarray
            Initial Euler angles in radians

        Returns
        -------
        np.ndarray
            Derivative of the euler angles

        """
        matrix = 1 / (np.cos(eul[1])) * np.array([
                            [np.cos(eul[1]), np.sin(eul[0]) * np.sin(eul[1]),  np.cos(eul[0]) * np.sin(eul[1])],  # noqa: E241, E501
                            [             0, np.cos(eul[0]) * np.cos(eul[1]), -np.sin(eul[0]) * np.cos(eul[1])],  # noqa: E201, E501
                            [             0,                  np.sin(eul[0]),                   np.cos(eul[0])]])  # noqa: E201, E241, E501
        vector = self.n / np.cos(eul) * np.array([
            np.sin(eul[2]),
            np.cos(eul[1]) * np.cos(eul[2]),
            np.sin(eul[1]) * np.sin(eul[2]),
        ])

        return matrix @ w + vector


class AttitudeQuat(Attitude):
    """Attitude representation using quaternions.

    Attributes
    ----------
    ang : ndarray
        Quaternions.
    w : ndarray
        Angular velocity vector (rad/s).
    """

    def __init__(self, initial_quat: np.ndarray):
        """Initialise attitude.

        Parameters
        ----------
        initial_quat: np.ndarray
            Initial quaternions

        """
        self.ang = initial_quat
        self.ang_vel = None

    def kinematic_diff_equation():  # noqa: ANN201, D102
        msg = "Kinematic differential equation not implemented for quaternions."
        raise NotImplementedError(msg)
