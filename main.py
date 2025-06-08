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
from attitudepy.blocks import (
    ClassicNDIControlLoop,
    INDIControlLoop,
    NDIModelBased,
    PIDController,
    TimescaleSeparationNDI,
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
eul_classic_control = False
eul_model_ndi = False
eul_timescale_ndi = False
eul_indi = False

quat_no_control_no_gravity = False
quat_no_control = True
quat_classic_control = False
quat_model_ndi = False
quat_timescale_ndi = False
quat_indi = False

tspan = [0, 1500]
tstep = 0.1
t = np.arange(tspan[0], tspan[1] + tstep, tstep)
integrator_settings = ScipyIntegrator(t)

# ###################################
# # PID controller gains
# ###################################

# Note that these have been tuned for the discrete control
# system, might be slightly different for the continous one.

# Gains for the classic control loop (Euler angles)
classic_kp = np.array([5, 5, 1])
classic_ki = np.array([0, 0, 0])
classic_kd = np.array([35, 35, 4])

# Gains for the NDI control loops (Euler angles)
ndi_kp = np.array([10, 10, 5])
ndi_ki = np.array([0, 0, 0])
ndi_kd = np.array([10, 10, 5])

# Gains for the NDI time scale separation control loop (Euler angles)
out_kp = np.array([3, 3, 3])
out_ki = np.array([0, 0, 0])
out_kd = np.array([3, 3, 3])

inn_kp = np.array([3, 3, 3])
inn_ki = np.array([0, 0, 0])
inn_kd = np.array([0, 0, 0])

# Gains for the classic control loop (quaternions angles)
classic_kp_quat = np.array([15, 15, 15, 0])
classic_ki_quat = np.array([0, 0, 0, 0])
classic_kd_quat = np.array([37, 37, 6, 0])

# Gains for the model NDI control loop (quaternions angles)
ndi_kp_quat = np.array([2, 2, 2, 0])
ndi_ki_quat = np.array([0, 0, 0, 0])
ndi_kd_quat = np.array([1.5, 1.5, 1.5, 0])

out_kp_quat = np.array([1.5, 1.5, 1.5, 0])
out_ki_quat = np.array([0, 0, 0, 0])
out_kd_quat = np.array([.7, .7, .7, 0])

inn_kp_quat = np.array([3, 3, 3])
inn_ki_quat = np.array([0, 0, 0])
inn_kd_quat = np.array([0, 0, 0])

# Block sample time
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
            0.0001, 0.0001, 0.0001,  # Nm
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
                ["Time (s)", "Angles (deg)"])

    plot_eul_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)"])

    plot_w_separate(time, state, ["$\\omega_1$ (deg/s)", "$\\omega_2$ (deg/s)", "$\\omega_3$ (deg/s)"],
                ["Time (s)", "Angular rates (deg/s)"])

if eul_classic_control:
    controller = PIDController(classic_kp, classic_ki, classic_kd, reference_commands,
                                sample_time=tsample)
    dynamics_simulator = initialise_euler(controller)

    print("Running Euler angles, classic control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)
    plt.savefig("images/eul/lin_control")


if eul_model_ndi:
    controller = PIDController(ndi_kp, ndi_ki, ndi_kd, reference_commands,
                                sample_time=tsample)
    control_loop = ClassicNDIControlLoop(controller)

    dynamics_simulator = initialise_euler(control_loop)

    print("Running Euler angles, NDI control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)


if eul_timescale_ndi:
    control_loop = TimescaleSeparationNDI(
        PIDController(out_kp, out_ki, out_kd, reference_commands, sample_time=tsample),
        PIDController(inn_kp, inn_ki, inn_kd, reference_commands, sample_time=tsample),
    )

    dynamics_simulator = initialise_euler(control_loop)

    print("Running Euler angles, timscale separation NDI control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)


if eul_indi:

    control_loop = INDIControlLoop(
        PIDController(out_kp, out_ki, out_kd, reference_commands, sample_time=tsample),
        PIDController(inn_kp, inn_ki, inn_kd, reference_commands, sample_time=tsample),
    )

    dynamics_simulator = initialise_euler(control_loop)

    print("Running Euler angles, incremental NDI control loop simulation")
    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_eul_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)


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

    plot_quat(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"])  # noqa: E501


if quat_no_control:
    dynamics_simulator = initialise_quat()

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], "Non-controlled attitude - Q")

    plot_quat_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"])
    plt.savefig("images/quat/no_control_ang")
    

    plot_w_separate(time, state, ["$\\omega_1$  (deg/s)", "$\\omega_2$ (deg/s)", "$\\omega_3$  (deg/s)"],
                ["Time (s)", "Angular rates (deg/s)"], from_quat=1)
    plt.savefig("images/quat/no_control_w")


if quat_classic_control:
    controller = PIDController(classic_kp_quat, classic_ki_quat, classic_kd_quat,
                                reference_commands_quat, sample_time=tsample)
    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    for i in range(len(state)):
        state[i, :4] /= np.linalg.norm(state[i, :4])

    plot_quat_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)

if quat_model_ndi:
    controller = PIDController(ndi_kp_quat, ndi_ki_quat, ndi_kd_quat, reference_commands_quat,
                                sample_time=tsample)
    control_loop = ClassicNDIControlLoop(controller)

    dynamics_simulator = initialise_quat(controller)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)

if quat_timescale_ndi:
    out_kp, out_ki, out_kd = out_kp_quat, out_ki_quat, out_kd_quat
    control_loop = TimescaleSeparationNDI(
        PIDController(out_kp, out_ki, out_kd, reference_commands_quat, sample_time=tsample),
        PIDController(inn_kp, inn_ki, inn_kd, reference_commands_quat, sample_time=tsample),
    )

    dynamics_simulator = initialise_quat(control_loop)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)


if quat_indi:
    out_kp, out_ki, out_kd = out_kp_quat, out_ki_quat, out_kd_quat
    control_loop = INDIControlLoop(
        PIDController(out_kp, out_ki, out_kd, reference_commands_quat, sample_time=tsample),
        PIDController(inn_kp, inn_ki, inn_kd, reference_commands_quat, sample_time=tsample),
    )

    dynamics_simulator = initialise_quat(control_loop)

    state_history = dynamics_simulator.simulate()
    time = np.array(list(state_history.keys()))
    state = np.vstack(list(state_history.values()))

    plot_quat_separate(time, state, ["$\\theta_1$ (deg)", "$\\theta_2$ (deg)", "$\\theta_3$ (deg)"],
                ["Time (s)", "Angles (deg)"], title=None,
                ref_function=reference_commands)


if np.any([
        eul_no_control_no_gravity,
        eul_no_control,
        eul_classic_control,
        eul_model_ndi,
        eul_timescale_ndi,
        eul_indi,
        quat_no_control_no_gravity,
        quat_no_control,
        quat_classic_control,
        quat_model_ndi,
        quat_timescale_ndi,
        quat_indi,
]):
    plt.show()
