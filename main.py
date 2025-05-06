"""Main script of the entire attitude dynamics & control loop."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from assignment_data import initialise_euler
from attitudepy import (
    PDController,
    dynamics_equation,
    dynamics_equation_nogravtorque,
)
from eulangles_plotting import plot_eul, plot_eul_separate

# ###################################
# # Main function options
# ###################################

no_control_no_gravity = True
no_control = True
classic_control = False

t = np.arange(0, 1500, 0.1)

classic_kp = [20, 20, 1]
classic_kd = [120, 120, 6.5]


def reference_commands(t, x):  # noqa: ANN001, ANN201, D103
    if t < 100:
        return np.array([0, 0, 0, 0, 0, 0]) * np.pi / 180

    if (t >= 100) and (t <= 500):
        return np.array([70, 70, 70, 0, 0, 0]) * np.pi / 180

    if (t > 500) and (t <= 900):
        return np.array([-70, -70, -70, 0, 0, 0]) * np.pi / 180

    return np.array([0, 0, 0, 0, 0, 0]) * np.pi / 180

# ###################################


if no_control_no_gravity:
    attitude, spacecraft = initialise_euler()

    y = odeint(dynamics_equation_nogravtorque, spacecraft.attitude.x0, t, (spacecraft,))

    plot_eul(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "No control, no gravity gradient torque")


if no_control:
    attitude, spacecraft = initialise_euler()

    y = odeint(dynamics_equation, spacecraft.attitude.x0, t, (spacecraft,))

    plot_eul(t, y, ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
                ["Time (s)", "Angles [deg]"], "Non-controlled attitude")


if np.any([no_control_no_gravity, no_control, classic_control]):
    plt.show()
