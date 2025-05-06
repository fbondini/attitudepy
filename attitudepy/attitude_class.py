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
    def kinematic_diff_equation(self, ang: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute the kinematic differential equation for attitude.

        Parameters
        ----------
        ang : ndarray
            Attitude vector (e.g., Euler angles or quaternion).
        w: np.ndarray
            Angular velocity in rad/s

        Returns
        -------
        ndarray
            Time derivative of the attitude vector.
        """
        return

    @abstractmethod
    def gravity_gradient_torque(self, ang: np.ndarray, n: np.ndarray,
                                        inertia: np.ndarray) -> np.ndarray:
        """Compute the torque given by gravity gradient.

        Parameters
        ----------
        ang : ndarray
            Attitude vector (e.g., Euler angles or quaternion).
        n: np.ndarray
            Orbital mean motion in rad/s.
        inertia: np.ndarray
            Inertia matrix.

        Returns
        -------
        ndarray
            Gravity gradient torque.
        """
        return

    @staticmethod
    def s_matrix(w: np.ndarray) -> np.ndarray:
        """S(w) matrix to transform cross product to matrix product.

        Parameters
        ----------
        w: np.ndarray
            Angular velocity (rad/s)

        Returns
        -------
        np.ndarray
            S matrix.
        """
        return np.array([
            [    0,  w[2], -w[1]],  # noqa: E201, E241
            [-w[2],     0,  w[0]],  # noqa: E241
            [ w[1], -w[0],     0],  # noqa: E201, E241
        ])


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
        self.w = np.array([0, 0, 0])
        self.x0 = np.append(self.ang, self.w)

    def kinematic_diff_equation(self, eul: np.ndarray, w: np.ndarray,  # noqa: PLR6301
                                        n: float) -> np.ndarray:
        """Define the kinematic differential equation.

        Gets the euler angles derivative, used in the dynamics to be integrated.

        Parameters
        ----------
        eul: np.ndarray
            Euler angles in radians
        w: np.ndarray
            Angular velocity in rad/s
        n: float
            Mean motion in rad/s

        Returns
        -------
        np.ndarray
            Derivative of the euler angles

        """
        matrix = 1 / (np.cos(eul[1])) * np.array([
                            [np.cos(eul[1]), np.sin(eul[0]) * np.sin(eul[1]),  np.cos(eul[0]) * np.sin(eul[1])],  # noqa: E241, E501
                            [             0, np.cos(eul[0]) * np.cos(eul[1]), -np.sin(eul[0]) * np.cos(eul[1])],  # noqa: E201, E501
                            [             0,                  np.sin(eul[0]),                   np.cos(eul[0])]])  # noqa: E201, E241, E501
        vector = n / np.cos(eul) * np.array([
            np.sin(eul[2]),
            np.cos(eul[1]) * np.cos(eul[2]),
            np.sin(eul[1]) * np.sin(eul[2]),
        ])

        return matrix @ w + vector

    def gravity_gradient_torque(self, eul: np.ndarray, n: np.ndarray,  # noqa: PLR6301
                                        inertia: np.ndarray) -> np.ndarray:
        """Compute the torque given by gravity gradient.

        Parameters
        ----------
        eul: np.ndarray
            Euler angles in radians
        n: np.ndarray
            Orbital mean motion in rad/s.
        inertia: np.ndarray
            Inertia matrix.

        Returns
        -------
        ndarray
            Gravity gradient torque.
        """
        left_matrix = np.array([
            [                               0, -np.cos(eul[0]) * np.cos(eul[1]), np.sin(eul[0]) * np.cos(eul[1])],  # noqa: E201, E501
            [ np.cos(eul[0]) * np.cos(eul[1]),                                0,                  np.sin(eul[1])],  # noqa: E201, E241, E501
            [-np.sin(eul[0]) * np.cos(eul[1]),                  -np.sin(eul[1]),                               0],  # noqa: E221, E241, E501
        ])

        right_vector = np.array([-np.sin(eul[1]), np.sin(eul[0]) * np.cos(eul[1]), np.cos(eul[0]) * np.cos(eul[1])])  # noqa: E501

        return 3 * n**2 * left_matrix @ inertia @ right_vector


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
        self.w = np.array([0, 0, 0])
        self.x0 = np.append(self.ang, self.w)

    def kinematic_diff_equation():  # noqa: ANN201, D102
        msg = "Kinematic differential equation not implemented for quaternions."
        raise NotImplementedError(msg)
