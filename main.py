"""Main script of the entire attitude dynamics & control loop."""

import numpy as np
from matplotlib import pyplot as plt

from assignment_data import initialise_euler, initialise_quat
from attitudepy import (
    AttitudeEuler,
    AttitudeQuat,
    Spacecraft,
    to_quat_state,
)
from attitudepy.controller import (
    NDIModelBased,
    NDITimeScaleSeparation,
    PDController,
)
from attitudepy.dynamics import DynamicsSimulatorNoGravityTorque
from attitudepy.integration import ScipyIntegrator
from plotting_utils import (
    plot_eul,
    plot_eul_separate,
    plot_quat,
    plot_quat_separate,
)

# ###################################
# # Main function options
# ###################################

eul_no_control_no_gravity = False
eul_no_control = False
eul_classic_control = True
eul_model_ndi = False
eul_timescale_ndi = False

quat_no_control_no_gravity = False
quat_no_control = False
quat_classic_control = True
quat_model_ndi = False
quat_timescale_ndi = False

tspan = [0, 1500]
tstep = 0.1
t = np.arange(tspan[0], tspan[1] + tstep, tstep)
integrator_settings = ScipyIntegrator(t)

classic_kp = [20, 20, 20]
classic_kd = [120, 120, 120]

classic_kp_quat = [60, 60, 60, 0]
classic_kd_quat = [120, 120, 120, 0]


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

    dynamics_simulator = DynamicsSimulatorNoGravityTorque(spacecraft,
                                                            integrator_settings)
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "No control, no gravity gradient torque - E")  # noqa: E501

if eul_no_control:
    dynamics_simulator = initialise_euler()

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Non-controlled attitude - E")


if eul_classic_control:
    controller = PDController(classic_kp, classic_kd, reference_commands)
    dynamics_simulator = initialise_euler(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Classically controlled attitude - E",
                reference_commands)


if eul_model_ndi:
    ndi = NDIModelBased()
    controller = PDController(classic_kp, classic_kd, reference_commands,
                                following=ndi)
    dynamics_simulator = initialise_euler(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Model based NDI controlled attitude - E",
                reference_commands)


if eul_timescale_ndi:
    ndi = NDITimeScaleSeparation()
    controller = PDController(classic_kp, classic_kd, reference_commands,
                                following=ndi)
    dynamics_simulator = initialise_euler(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Time Scale Sep. NDI controlled attitude - E",  # noqa: E501
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

    dynamics_simulator = DynamicsSimulatorNoGravityTorque(spacecraft,
                                                            integrator_settings)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "No control, no gravity gradient torque - Q")  # noqa: E501


if quat_no_control:
    dynamics_simulator = initialise_quat()

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Non-controlled attitude - Q")


if quat_classic_control:
    controller = PDController(classic_kp_quat, classic_kd_quat, reference_commands_quat)
    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    for i in range(len(state)):
        state[i, :4] /= np.linalg.norm(state[i, :4])

    plot_quat_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Classically controlled attitude - Q",
                reference_commands)


if quat_model_ndi:
    ndi = NDIModelBased()
    controller = PDController(classic_kp_quat, classic_kd_quat, reference_commands_quat,
                                following=ndi)
    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Model based NDI controlled attitude - Q",
                reference_commands)


if quat_timescale_ndi:
    ndi = NDITimeScaleSeparation()
    controller = PDController(classic_kp_quat, classic_kd_quat, reference_commands_quat,
                                following=ndi)
    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Time Scale Sep. NDI controlled attitude - Q",  # noqa: E501
                reference_commands)


if np.any([
        eul_no_control_no_gravity,
        eul_no_control,
        eul_classic_control,
        eul_model_ndi,
        eul_timescale_ndi,
        quat_no_control_no_gravity,
        quat_no_control,
        quat_classic_control,
        quat_model_ndi,
        quat_timescale_ndi,
]):
    plt.show()
