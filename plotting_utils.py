"""Euler angles plotting function."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

from attitudepy import to_euler


def plot_eul(t: np.ndarray, eul: np.ndarray, labels: List,
                axislabels: List, title: Optional[str] = None,
                ) -> None:
    """Plot euler angles with provided settings."""
    style.use("default.mplstyle")

    plt.figure()

    plt.plot(t, eul[:, 0] * 180 / np.pi, label=labels[0])
    plt.plot(t, eul[:, 1] * 180 / np.pi, label=labels[1])
    plt.plot(t, eul[:, 2] * 180 / np.pi, label=labels[2])

    plt.grid(True)  # noqa: FBT003
    plt.legend(loc="best")
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    if title is not None:
        plt.title(title)
    plt.tight_layout()


def plot_eul_separate(t: np.ndarray, y: np.ndarray, labels: List[str],
                        axislabels: List[str], title: Optional[str] = None,
                        ref_function: Optional[callable] = None) -> None:
    """Plot each Euler angle and its reference on separate subplots in one row."""
    style.use("default.mplstyle")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    if ref_function is not None:
        reference = ref_function(t[0], y[0])
        for i, ti in enumerate(t[1:]):
            reference = np.vstack([reference, ref_function(ti, y[i])])
    for i in range(3):
        angle_deg = y[:, i] * 180 / np.pi
        axs[i].plot(t, angle_deg, label=labels[i])

        if ref_function is not None:
            ref_deg = reference[:, i] * 180 / np.pi
            axs[i].plot(t, ref_deg, label=f"Ref {labels[i]}", linestyle="--")

        axs[i].grid(True)  # noqa: FBT003
        axs[i].set_xlabel(axislabels[0])
        axs[i].set_ylabel(axislabels[1])
        axs[i].legend(loc="best")
        axs[i].set_title(f"{labels[i]} vs Time")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()


def plot_w(t: np.ndarray, w: np.ndarray, labels: List,
                axislabels: List, title: Optional[str] = None,
                ) -> None:
    """Plot angular velocity with provided settings."""
    style.use("default.mplstyle")

    plt.figure()

    plt.plot(t, w[:, 0] * 180 / np.pi, label=labels[0])
    plt.plot(t, w[:, 1] * 180 / np.pi, label=labels[1])
    plt.plot(t, w[:, 2] * 180 / np.pi, label=labels[2])

    plt.grid(True)  # noqa: FBT003
    plt.legend(loc="best")
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    if title is not None:
        plt.title(title)
    plt.tight_layout()


def plot_w_separate(t: np.ndarray, y: np.ndarray, labels: List[str],
                        axislabels: List[str], title: Optional[str] = None,
                        from_quat: int = 0) -> None:
    """Plot each Euler angle and its reference on separate subplots in one row."""
    style.use("default.mplstyle")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    for i in range(3):
        angle_deg = y[:, 3 + from_quat + i] * 180 / np.pi
        axs[i].plot(t, angle_deg, label=labels[i])

        axs[i].grid(True)  # noqa: FBT003
        axs[i].set_xlabel(axislabels[0])
        axs[i].set_ylabel(axislabels[1])
        axs[i].legend(loc="best")
        axs[i].set_title(f"{labels[i]} vs Time")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()


def plot_quat(t: np.ndarray, q: np.ndarray, labels: List,
                axislabels: List, title: Optional[str] = None,
                ) -> None:
    """Plot quaternions state as eul angles with provided settings."""
    style.use("default.mplstyle")

    eul = to_euler(q[0, :4])
    for i, _ in enumerate(t[1:]):
        eul = np.vstack([eul, to_euler(q[i, :4])])

    plt.figure()

    plt.plot(t, eul[:, 0] * 180 / np.pi, label=labels[0])
    plt.plot(t, eul[:, 1] * 180 / np.pi, label=labels[1])
    plt.plot(t, eul[:, 2] * 180 / np.pi, label=labels[2])

    plt.grid(True)  # noqa: FBT003
    plt.legend(loc="best")
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    if title is not None:
        plt.title(title)
    plt.tight_layout()


def plot_quat_separate(t: np.ndarray, q: np.ndarray, labels: List[str],
                        axislabels: List[str], title: Optional[str] = None,
                        ref_function: Optional[callable] = None) -> None:
    """Plot quaternions state as eul angles and its reference on subplots in one row."""
    style.use("default.mplstyle")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    if ref_function is not None:
        reference = ref_function(t[0], q[0, :4])
        for i, ti in enumerate(t[1:]):
            reference = np.vstack([reference, ref_function(ti, q[i, :4])])

    eul = to_euler(q[0, :4])
    for i, _ in enumerate(t[1:]):
        eul = np.vstack([eul, to_euler(q[i, :4])])

    for i in range(3):
        angle_deg = eul[:, i] * 180 / np.pi
        axs[i].plot(t, angle_deg, label=labels[i])

        if ref_function is not None:
            ref_deg = reference[:, i] * 180 / np.pi
            axs[i].plot(t, ref_deg, label=f"Ref {labels[i]}", linestyle="--")

        axs[i].grid(True)  # noqa: FBT003
        axs[i].set_xlabel(axislabels[0])
        axs[i].set_ylabel(axislabels[1])
        axs[i].legend(loc="best")
        axs[i].set_title(f"{labels[i]} vs Time")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
