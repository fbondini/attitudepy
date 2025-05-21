"""Attitude class."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

ORDER = "xyz"
EUL_LENGTH_ERROR_MSG = "The euler angles vector must have length 3"
QUAT_LENGTH_ERROR_MSG = "The quaternion vector must have length 4"


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

    @property
    def x(self) -> np.ndarray:
        """Return the state vector.

        Returns
        -------
        ndarray
            Combination of the state vector.
        """
        return np.append(self.ang, self.w)

    def kinematic_diff_equation(self, n: float) -> np.ndarray:
        """Compute the kinematic differential equation for attitude.

        Parameters
        ----------
        n: float
            Orbit mean motion

        Returns
        -------
        ndarray
            Time derivative of the attitude vector.
        """
        ang = self.ang
        w = self.w

        c2 = self.c_matrix()[:, 1]
        if len(ang) == 4:
            w = np.append(w, 0)
            c2 = np.append(c2, 0)

        return self.w2angdot_matrix() @ (w + n * c2)

    def gravity_gradient_torque(self, n: np.ndarray,
                                        inertia: np.ndarray) -> np.ndarray:
        """Compute the torque given by gravity gradient.

        Parameters
        ----------
        n: np.ndarray
            Orbital mean motion in rad/s.
        inertia: np.ndarray
            Inertia matrix.

        Returns
        -------
        ndarray
            Gravity gradient torque.
        """
        c3 = self.c_matrix()[:, 2]
        return np.cross(3 * n**2 * c3, inertia @ c3)

    @abstractmethod
    def state_error(self, ref_ang: np.ndarray, ref_w: np.ndarray,
                        n: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the error between reference and current state.

        Parameters
        ----------
        ref_ang : ndarray
            Reference attitude vector (e.g., Euler angles or quaternion).
        ref_w: np.ndarray
            Reference angular velocity in rad/s
        n: float
            Orbit mean motion in rad/s

        Returns
        -------
        e, e_dot: ndarray, ndarray
            Error and error derivative to be passed to the Block u function.
            .
        """
        return

    def s_matrix(self) -> np.ndarray:
        """S(w) matrix to transform cross product to matrix product.

        Returns
        -------
        np.ndarray
            S matrix.
        """
        w = self.w
        return np.array([
            [    0,  w[2], -w[1]],  # noqa: E201, E241
            [-w[2],     0,  w[0]],  # noqa: E241
            [ w[1], -w[0],     0],  # noqa: E201, E241
        ])

    @abstractmethod
    def c_matrix(self) -> np.ndarray:
        """C(ang) matrix to express rotations of attitude angles.

        Returns
        -------
        ndarray
            C matrix.
        """
        return

    @abstractmethod
    def w2angdot_matrix(self) -> np.ndarray:
        """Matrix to transform from angular velocity to attitude derivative.

        Returns
        -------
        ndarray
            Angular velocity -> attitude derivative, matrix.
        """
        return

    @property
    @abstractmethod
    def nwdx_matrix(self) -> np.ndarray:
        """Compute the derivative with respect to the state of N*w.

        Where N is w2angdot and w is the angular velocity.

        Returns
        -------
        ndarray
            Derivative with respect to the state of N*w.
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

        Raises
        ------
        ValueError
            If the input angles are not a vector of length 3.

        """
        if len(initial_eul_angles) != 3:
            raise ValueError(EUL_LENGTH_ERROR_MSG)

        self.ang = initial_eul_angles
        self.w = np.array([0, 0, 0])

    def state_error(self, ref_ang: np.ndarray,
                        ref_thetadot: np.ndarray,
                        n: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the error between reference and current state.

        Parameters
        ----------
        ref_ang : ndarray
            Reference Euler angles.
        ref_thetadot: np.ndarray
            Reference derivative of the euler angles rad/s
        n: float
            Orbit mean motion in rad/s

        Returns
        -------
        e, e_dot: ndarray, ndarray
            Error and error derivative to be passed to the Block u function.
            .
        """
        e = self.ang - ref_ang
        e_dot = self.kinematic_diff_equation(n) - ref_thetadot

        return e, e_dot

    def c_matrix(self) -> np.ndarray:
        """C(ang) matrix to express rotations of attitude angles.

        Returns
        -------
        ndarray
            C matrix.
        """
        eul = self.ang
        return np.array([
            [np.cos(eul[1]) * np.cos(eul[2]), np.cos(eul[1]) * np.sin(eul[2]), -np.sin(eul[1])],  # noqa: E501
            [np.cos(eul[2]) * np.sin(eul[0]) * np.sin(eul[1]) - np.cos(eul[0]) * np.sin(eul[2]), np.cos(eul[0]) * np.cos(eul[2]) + np.sin(eul[0]) * np.sin(eul[1]) * np.sin(eul[2]), np.cos(eul[1]) * np.sin(eul[0])],  # noqa: E501
            [np.sin(eul[0]) * np.sin(eul[2]) + np.cos(eul[0]) * np.cos(eul[2]) * np.sin(eul[1]), np.cos(eul[0]) * np.sin(eul[1]) * np.sin(eul[2]) - np.cos(eul[2]) * np.sin(eul[0]), np.cos(eul[0]) * np.cos(eul[1])],  # noqa: E501
        ])

    def w2angdot_matrix(self) -> np.ndarray:
        """Matrix to transform from angular velocity to euler angle derivative.

        Returns
        -------
        ndarray
            Angular velocity -> euler derivative, matrix.
        """
        eul = self.ang
        return 1 / (np.cos(eul[1])) * np.array([
                            [np.cos(eul[1]), np.sin(eul[0]) * np.sin(eul[1]),  np.cos(eul[0]) * np.sin(eul[1])],  # noqa: E241, E501
                            [             0, np.cos(eul[0]) * np.cos(eul[1]), -np.sin(eul[0]) * np.cos(eul[1])],  # noqa: E201, E501
                            [             0,                  np.sin(eul[0]),                   np.cos(eul[0])]])  # noqa: E201, E241, E501

    @property
    def nwdx_matrix(self) -> np.ndarray:
        """Compute the derivative with respect to the state of N*w.

        Where N is w2angdot and w is the angular velocity.

        Returns
        -------
        ndarray
            Derivative with respect to the state of N*w.
        """
        theta1, theta2, _ = self.ang
        _, w2, w3 = self.w
        left_hand_side = 0.5 * np.array([
            [(np.cos(theta1)*w2 - np.sin(theta1)*w3)*np.tan(theta2), (np.sin(theta1)*w2 + np.cos(theta1)*w3)/np.cos(theta2)**2, 0],  # noqa: E226, E501
            [-np.sin(theta1)*w2 -np.cos(theta1)*w3, 0, 0],  # noqa: E226
            [(np.cos(theta1)*w2 - np.sin(theta1)*w3)/np.cos(theta2), (np.sin(theta1)*w2 + np.cos(theta1)*w3)*np.tan(theta2)/np.cos(theta2), 0],  # noqa: E226, E501
        ])
        return np.column_stack([left_hand_side, self.w2angdot_matrix()])

    def to_quat(self) -> "AttitudeQuat":
        """Convert Euler angles to quaternion representation.

        Returns
        -------
        AttitudeQuat
            Equivalent attitude in quaternion representation.
        """
        return AttitudeQuat(to_quat(self.ang))


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

        Raises
        ------
        ValueError
            If the input angles are not a vector of length 3.

        """
        if len(initial_quat) != 4:
            raise ValueError(QUAT_LENGTH_ERROR_MSG)

        self.ang = initial_quat
        self.w = np.array([0, 0, 0])

    def c_matrix(self) -> np.ndarray:
        """C(ang) matrix to express rotations of attitude angles.

        Returns
        -------
        ndarray
            C matrix.
        """
        q = self.ang
        return np.array([
            [    1 - 2 * (q[1]**2 + q[2]**2), 2 * (q[0] * q[1] + q[2] * q[3]), 2 * (q[0] * q[2] - q[1] * q[3])],  # noqa: E201, E501
            [2 * (q[0] * q[1] - q[2] * q[3]),     1 - 2 * (q[0]**2 + q[2]**2), 2 * (q[1] * q[2] + q[0] * q[3])],  # noqa: E241, E501
            [2 * (q[0] * q[2] + q[1] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]),     1 - 2 * (q[0]**2 + q[1]**2)],  # noqa: E241, E501
        ])

    def state_error(self, ref_ang: np.ndarray,
                        ref_w: np.ndarray,
                        n: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the error between reference and current state.

        Parameters
        ----------
        ref_ang : ndarray
            Reference quaternions.
        ref_w: np.ndarray
            Reference angular velocity in rad/s
        n: float
            Orbit mean motion in rad/s

        Returns
        -------
        e, e_dot: ndarray, ndarray
            Error and error derivative to be passed to the Block u function.
            .
        """
        current_ang = self.ang
        current_w = self.w
        e = self._q_rot_matrix(ref_ang) @ current_ang
        e_dot = current_w - ref_w

        return e, e_dot

    @staticmethod
    def _q_rot_matrix(q: np.ndarray) -> np.ndarray:
        return np.array([
            [q[3], q[2], -q[1], -q[0]],
            [-q[2], q[3], q[0], -q[1]],
            [q[1], -q[0], q[3], -q[2]],
            [q[0], q[1], q[2], q[3]],
        ])

    def w2angdot_matrix(self) -> np.ndarray:
        """Matrix to transform from angular velocity to quaternion derivative.

        Returns
        -------
        ndarray
            Angular velocity -> quaternions, matrix.
        """
        q = self.ang
        return 0.5 * self._q_rot_matrix(q).T

    @property
    def nwdx_matrix(self) -> np.ndarray:
        """Compute the derivative with respect to the state of N*w.

        Where N is w2angdot and w is the angular velocity.

        Returns
        -------
        ndarray
            Derivative with respect to the state of N*w.
        """
        left_hand_side = 0.5 * np.array([
            [0, -self.w[2], self.w[1], -self.w[0]],
            [self.w[2], 0, -self.w[0], -self.w[1]],
            [-self.w[1], self.w[0], 0, -self.w[2]],
            [self.w[0], self.w[1], self.w[2], 0],
        ])
        return np.column_stack([left_hand_side, self.w2angdot_matrix()[:, :3]])

    def to_euler(self) -> AttitudeEuler:
        """Convert quaternion to Euler angle representation.

        Returns
        -------
        AttitudeEuler
            Equivalent attitude in Euler angle representation.
        """
        return AttitudeEuler(to_euler(self.ang))


def to_quat(eul: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion representation.

    Parameters
    ----------
    eul: ndarray
        Euler angles.

    Returns
    -------
    ndarray
        Equivalent attitude in quaternion representation.
    """
    rotation = Rotation.from_euler(ORDER, eul)
    return rotation.as_quat()  # Returns [x, y, z, w] where w is cos(phi/2)


def to_quat_state(state: np.ndarray) -> np.ndarray:
    """Convert Euler angles state to quaternion state.

    Parameters
    ----------
    state: ndarray
        [eul1, eul2, eul3, w1, w2, w3] in rad and rad/s.
        Where euli is the i-th component of the euler angle attitude
        and wi the i-th component of the angular velocity.

    Returns
    -------
    ndarray
        Equivalent state in quaternion representation.
    """
    return np.append(to_quat(state[:3]), state[3:])


def to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angle representation.

    Parameters
    ----------
    quat: ndarray
        Quaternions.

    Returns
    -------
    ndarray
        Equivalent attitude in Euler angle representation.
    """
    rotation = Rotation.from_quat(quat)  # with q4 = cos(phi/2)
    return rotation.as_euler(ORDER)  # Returns in radians


def to_euler_state(state: np.ndarray) -> np.ndarray:
    """Convert quaternions state to Euler angle state.

    Parameters
    ----------
    state: ndarray
        [q1, q2, q3, q4, w1, w2, w3] in - and rad/s.
        Where qi is the i-th component of the quaternions
        and wi the i-th component of the angular velocity.

    Returns
    -------
    ndarray
        Equivalent state in Euler angles representation.
    """
    return np.append(to_euler(state[:4]), state[4:])
