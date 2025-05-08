"""Initialise assignment data."""
import numpy as np

from attitudepy import (
    AttitudeEuler,
    AttitudeQuat,
    Controller,
    DynamicsSimulator,
    Spacecraft,
    to_quat,
)


def initialise_euler(controller: Controller = None) -> DynamicsSimulator:
    """Initialise attitude and spacecraft with the assignment data."""  # noqa: DOC201
    attitude = AttitudeEuler(
            np.array([30, 30, 30]) * np.pi / 180,
        )

    spacecraft = Spacecraft(
        initial_attitude=attitude,
        orbit_alt=700,  # km
        inertia=np.array([
            [124.531,       0,     0],  # noqa: E241
            [      0, 124.586,     0],  # noqa: E201, E241
            [      0,       0, 0.704],  # noqa: E201, E241
        ]),
        torque_disturb=np.array([
            0.001, 0.001, 0.001,  # Nm
        ]),
    )

    return DynamicsSimulator(spacecraft, controller)


def initialise_quat(controller: Controller = None) -> DynamicsSimulator:
    """Initialise attitude and spacecraft with the assignment data."""  # noqa: DOC201
    attitude = AttitudeQuat(
            to_quat(np.array([30, 30, 30]) * np.pi / 180),
        )

    spacecraft = Spacecraft(
        initial_attitude=attitude,
        orbit_alt=700,  # km
        inertia=np.array([
            [124.531,       0,     0],  # noqa: E241
            [      0, 124.586,     0],  # noqa: E201, E241
            [      0,       0, 0.704],  # noqa: E201, E241
        ]),
        torque_disturb=np.array([
            0.001, 0.001, 0.001,  # Nm
        ]),
    )

    return DynamicsSimulator(spacecraft, controller)
