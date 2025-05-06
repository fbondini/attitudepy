"""Attitude dynamics functions."""
import numpy as np

from .controller import Controller
from .spacecraft_class import Spacecraft


# FUTURE: refactor with control inside of spacecraft
def dynamics_equation(x: np.ndarray, t: float,
                            sc: Spacecraft, ctrl: Controller = None) -> np.ndarray:
    """Define the dynamics differential equations.

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
    ctrl: Controller (optional)
        Controller object

    Returns
    -------
    xdot: float
        Attitude state derivative.
    """
    xdot = dynamics_equation_nogravtorque(x, t, sc, ctrl)
    xdot[-3:] = xdot[-3:] + (np.linalg.inv(sc.inertia) @
            sc.attitude.gravity_gradient_torque(x[0:-3], sc.mean_motion, sc.inertia))

    return xdot


def dynamics_equation_nogravtorque(x: np.ndarray, t: float,
                            sc: Spacecraft, ctrl: Controller = None) -> np.ndarray:
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
    ctrl: Controller (optional)
        Controller object

    Returns
    -------
    xdot: float
        Attitude state derivative.
    """
    ang = x[0:-3]
    w = x[-3:]

    angdot = sc.attitude.kinematic_diff_equation(ang, w, sc.mean_motion)

    fx_part = np.linalg.inv(sc.inertia) @ sc.attitude.s_matrix(w) @ sc.inertia @ w

    # Control and disturbance torques
    if ctrl is not None:
        ref = ctrl.guidance(t, x)
        e, e_dot = sc.attitude.state_error(ang, w, ref[:-3], ref[-3:], sc.mean_motion)
        u = ctrl.u(e, e_dot) + sc.torque_disturb
    else:
        u = np.zeros(3)

    gu_part = np.linalg.inv(sc.inertia) @ u

    return np.append(angdot, fx_part + gu_part)
