"""Main script of the entire attitude dynamics & control loop."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from scipy.integrate import odeint

from attitudepy import AttitudeEuler, Spacecraft, dynamics_equation

style.use("default.mplstyle")

attitude = AttitudeEuler(
    np.array([30, 30, 30]) * np.pi / 180,
)

sc = Spacecraft(
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

t = np.arange(0, 1500, 0.1)
y = odeint(dynamics_equation, sc.attitude.x0, t, (sc,))

plt.figure()

plt.plot(t, y[:, 0] * 180 / np.pi, label="$\\theta_1$")
plt.plot(t, y[:, 1] * 180 / np.pi, label="$\\theta_2$")
plt.plot(t, y[:, 2] * 180 / np.pi, label="$\\theta_3$")

plt.grid(True)  # noqa: FBT003
plt.legend(loc="best")
plt.xlabel("Time (s)")
plt.ylabel("Euler angles (deg)")
plt.title("Attitude without control")
plt.tight_layout()

plt.show()
