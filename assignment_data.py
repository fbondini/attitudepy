"""Initialise assignment data."""
import numpy as np

from attitudepy import (
    AttitudeEuler,
    AttitudeQuat,
    Spacecraft,
    to_quat,
)
from attitudepy.blocks import Block
from attitudepy.dynamics import DynamicsSimulator
from attitudepy.integration import ScipyIntegrator


def initialise_euler(controller: Block = None) -> DynamicsSimulator:
    """Initialise attitude and spacecraft with the assignment data."""  # noqa: DOC201
    tspan = [0, 1500]
    tstep = 0.1
    t = np.arange(tspan[0], tspan[1] + tstep, tstep)
    integrator_settings = ScipyIntegrator(t)

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
            0.0001, 0.0001, 0.0001,  # Nm
        ]),
    )

    return DynamicsSimulator(spacecraft, integrator_settings, controller)


def initialise_quat(controller: Block = None) -> DynamicsSimulator:
    """Initialise attitude and spacecraft with the assignment data."""  # noqa: DOC201
    tspan = [0, 1500]
    tstep = 0.1
    t = np.arange(tspan[0], tspan[1] + tstep, tstep)
    integrator_settings = ScipyIntegrator(t)

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
            0.0001, 0.0001, 0.0001,  # Nm
        ]),
    )

    return DynamicsSimulator(spacecraft, integrator_settings, controller)
