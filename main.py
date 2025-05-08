"""Main script of the entire attitude dynamics & control loop."""

import numpy as np
from matplotlib import pyplot as plt

from assignment_data import initialise_euler, initialise_quat
from attitudepy import (
    AttitudeEuler,
    AttitudeQuat,
    DynamicsSimulatorNoGravityTorque,
    PDController,
    Spacecraft,
    to_quat_state,
)
from plotting_utils import plot_eul, plot_eul_separate, plot_quat, plot_quat_separate

# ###################################
# # Main function options
# ###################################

eul_no_control_no_gravity = True
eul_no_control = True
eul_classic_control = True

quat_no_control_no_gravity = True
quat_no_control = True
quat_classic_control = True

t = np.arange(0, 1500, 0.1)

classic_kp = [20, 20, 1]
classic_kd = [120, 120, 6.5]

classic_kp_quat = [500, 500, 20]
classic_kd_quat = [300, 300, 10]


def reference_commands(t, x):  # noqa: ANN001, ANN201, D103
    if t < 100:
        return np.array([0, 0, 0, 0, 0, 0]) * np.pi / 180

    if (t >= 100) and (t <= 500):
        return np.array([70, 70, 70, 0, 0, 0]) * np.pi / 180

    if (t > 500) and (t <= 900):
        return np.array([-70, -70, -70, 0, 0, 0]) * np.pi / 180

    return np.array([0, 0, 0, 0, 0, 0]) * np.pi / 180


def reference_commands_quat(t, x):  # noqa: ANN001, ANN201, D103
    return to_quat_state(reference_commands(t, x))

# ###################################


if eul_no_control_no_gravity:
    attitude = AttitudeEuler(
            np.array([0, 0, 0]) * np.pi / 180,
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

    dynamics_simulator = DynamicsSimulatorNoGravityTorque(spacecraft)
    y = dynamics_simulator.simulate(t)

    plot_eul(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "No control, no gravity gradient torque - E")  # noqa: E501


if eul_no_control:
    dynamics_simulator = initialise_euler()
    y = dynamics_simulator.simulate(t)

    plot_eul(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "Non-controlled attitude - E")


if eul_classic_control:
    controller = PDController(classic_kp, classic_kd, reference_commands)
    dynamics_simulator = initialise_euler(controller)

    y = dynamics_simulator.simulate(t)

    plot_eul_separate(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "Classically controlled attitude - E",
                reference_commands)


if quat_no_control_no_gravity:
    attitude = AttitudeQuat(
            np.array([0, 0, 0, 1]),
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

    dynamics_simulator = DynamicsSimulatorNoGravityTorque(spacecraft)
    y = dynamics_simulator.simulate(t)

    plot_quat(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "No control, no gravity gradient torque - Q")  # noqa: E501


if quat_no_control:
    dynamics_simulator = initialise_quat()

    y = dynamics_simulator.simulate(t)

    plot_quat(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "Non-controlled attitude - Q")


if quat_classic_control:
    controller = PDController(classic_kp_quat, classic_kd_quat, reference_commands_quat)
    dynamics_simulator = initialise_quat(controller)

    y = dynamics_simulator.simulate(t)

    for i in range(len(y)):
        y[i, :4] /= np.linalg.norm(y[i, :4])

    plot_quat_separate(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "Classically controlled attitude - Q",
                reference_commands)


if np.any([
        eul_no_control_no_gravity,
        eul_no_control,
        eul_classic_control,
        quat_no_control_no_gravity,
        quat_no_control,
        quat_classic_control,
]):
    plt.show()
