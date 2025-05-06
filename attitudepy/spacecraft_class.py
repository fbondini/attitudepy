"""Spacecraft class."""
from dataclasses import dataclass

import numpy as np

from .attitude_class import Attitude

GRAV_PARAMETER_EARTH = 398600.4
RADIUS_EARTH = 6378  # km


@dataclass
class Spacecraft:
    """Class to store Spacecraft related data.

    Attributes
    ----------
    attitude: Attitude
        Current attitude of the spacecraft
    orbit_alt: float
        Orbit altitude in km (assumes circular orbit)
    mean_motion: float
        Orbital mean motion in rad/s
    inertia: np.ndarray
        Spacecraft inertia matrix
    torque_disturb: np.ndarray
        Disturbance in the control torques
    """

    def __init__(self, initial_attitude: Attitude, orbit_alt: float,
                        inertia: np.ndarray, torque_disturb: np.ndarray):
        """Initialise the spacecraft.

        Parameters
        ----------
        attitude: Attitude
            Current attitude of the spacecraft
        orbit_alt: float
            Orbit altitude in km (assumes circular orbit)
        inertia: np.ndarray
            Spacecraft inertia matrix
        torque_disturb: np.ndarray
            Disturbance in the control torques
        """
        self.attitude = initial_attitude
        self.orbit_alt = orbit_alt
        self.mean_motion = np.sqrt(
            GRAV_PARAMETER_EARTH / (self.orbit_alt + RADIUS_EARTH)**3)
        self.inertia = inertia
        self.torque_disturb = torque_disturb
