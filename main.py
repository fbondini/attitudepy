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
    PIDController,
)
from attitudepy.dynamics import DynamicsSimulatorNoGravityTorque
from attitudepy.integration import ScipyIntegrator
from plotting_utils import (
    plot_eul,
    plot_eul_separate,
    plot_quat,
    plot_quat_separate,
    plot_w_separate,
)

# ###################################
# # Main function options
# ###################################

eul_no_control_no_gravity = False
eul_no_control = False
eul_classic_control = True
eul_model_ndi = True
eul_timescale_ndi = False

quat_no_control_no_gravity = False
quat_no_control = False
quat_classic_control = False
quat_model_ndi = False
quat_timescale_ndi = False

tspan = [0, 1500]
tstep = 0.01
t = np.arange(tspan[0], tspan[1] + tstep, tstep)
integrator_settings = ScipyIntegrator(t)

# ###################################
# # PID controller gains
# ###################################

# Note that these have been tuned for the discrete control
# system, might be slightly different for the continous one.

# Gains for the classic control loop (Euler angles)
classic_kp = np.array([10, 5, 0.5])
classic_ki = np.array([0, 0, 0])
classic_kd = np.array([70, 70, 2])

# Gains for the NDI control loops (Euler angles)
ndi_kp = np.array([1, 1, 1])
ndi_ki = np.array([0, 0, 0])
ndi_kd = np.array([2, 2, 2])

# Gains for the classic control loop (quaternions angles)
classic_kp_quat = np.array([10, 5, 0.5, 0])
classic_ki_quat = np.array([0, 0, 0, 0])
classic_kd_quat = np.array([50, 50, 1.19, 0])

# Controller sample time
tsample = 0.1


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

    print("Running Euler angles, no control, no distrubance simulation")
    dynamics_simulator = DynamicsSimulatorNoGravityTorque(spacecraft,
                                                            integrator_settings)
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "No control, no gravity gradient torque - E")  # noqa: E501

if eul_no_control:
    dynamics_simulator = initialise_euler()

    print("Running Euler angles, no control simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Non-controlled attitude - E")

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "No control - E")

    plot_w_separate(time, state, ["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"],
                ["Time (s)", "Angular rates (deg/s)"], "No control - E")


if eul_classic_control:
    controller = PIDController(classic_kp, classic_ki, classic_kd, reference_commands,
                                sample_time=tsample)
    dynamics_simulator = initialise_euler(controller)

    print("Running Euler angles, classic control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Classically controlled attitude - E",
                reference_commands)


if eul_model_ndi:
    ndi = NDIModelBased()
    controller = PIDController(ndi_kp, ndi_ki, ndi_kd, reference_commands,
                                following=ndi, sample_time=tsample)
    dynamics_simulator = initialise_euler(controller)

    print("Running Euler angles, NDI control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Model based NDI controlled attitude - E",
                reference_commands)


if eul_timescale_ndi:
    ndi = NDITimeScaleSeparation()
    controller = PIDController(classic_kp, classic_ki, classic_kd, reference_commands,
                                following=ndi, sample_time=tsample)
    dynamics_simulator = initialise_euler(controller)

    print("Running Euler angles, timscale separation NDI control loop simulation")
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

    plot_quat_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "No control - Q")

    plot_w_separate(time, state, ["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"],
                ["Time (s)", "Angular rates (deg/s)"], "No control - Q", from_quat=1)


if quat_classic_control:
    controller = PIDController(classic_kp_quat, classic_ki_quat, classic_kd_quat,
                                reference_commands_quat, sample_time=tsample)
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
    controller = PIDController(classic_kp_quat, classic_ki_quat, classic_kd_quat,
                                reference_commands_quat,
                                following=ndi, sample_time=tsample)
    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles (deg)"], "Model based NDI controlled attitude - Q",
                reference_commands)


if quat_timescale_ndi:
    ndi = NDITimeScaleSeparation()
    controller = PIDController(classic_kp_quat, classic_ki_quat, classic_kd_quat,
                                reference_commands_quat,
                                following=ndi, sample_time=tsample)
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
