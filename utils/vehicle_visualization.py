#!/usr/bin/env python3
"""Visualization helpers for integrated vehicle dynamics simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(x: np.ndarray, y: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(x, y, lw=2.0)
    ax.set_title("Vehicle Trajectory")
    ax.set_xlabel("Global X [m]")
    ax.set_ylabel("Global Y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gg_diagram(ax_long_g: np.ndarray, ay_lat_g: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(ax_long_g, ay_lat_g, s=8, alpha=0.45)
    ax.set_title("G-G Diagram")
    ax.set_xlabel("Longitudinal G")
    ax.set_ylabel("Lateral G")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_speed_trace(time_s: np.ndarray, speed_kmh: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(time_s, speed_kmh, lw=2.0)
    ax.set_title("Speed Trace")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tire_forces(time_s: np.ndarray, tire_forces: np.ndarray) -> plt.Figure:
    """Plot longitudinal and lateral force histories for 4 tires.

    tire_forces shape: [N, 4, 2] with last axis [Fx, Fy].
    """
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True)
    labels = ["FL", "FR", "RL", "RR"]
    for i in range(4):
        axes[0].plot(time_s, tire_forces[:, i, 0], lw=1.5, label=labels[i])
        axes[1].plot(time_s, tire_forces[:, i, 1], lw=1.5, label=labels[i])

    axes[0].set_ylabel("Fx [N]")
    axes[1].set_ylabel("Fy [N]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("Tire Force History")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(ncol=4)
    fig.tight_layout()
    return fig


def plot_load_transfer(time_s: np.ndarray, normal_loads: np.ndarray) -> plt.Figure:
    """Visualize normal loads for FL/FR/RL/RR over time.

    normal_loads shape: [N, 4].
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    labels = ["FL", "FR", "RL", "RR"]
    for i in range(4):
        ax.plot(time_s, normal_loads[:, i], lw=1.7, label=labels[i])

    ax.set_title("Load Transfer")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normal Load [N]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    return fig
