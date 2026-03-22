#!/usr/bin/env python3
"""Visualization helpers for active aero behavior."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from vehicle_dynamics.aerodynamics import AerodynamicsModel


def _sweep_forces(
    model: AerodynamicsModel,
    speeds_kmh: np.ndarray,
    mode: str,
    ride_height_front: float = 0.05,
    ride_height_rear: float = 0.055,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    downforce_front = []
    downforce_rear = []
    drag = []
    for speed_kmh in speeds_kmh:
        speed = speed_kmh / 3.6
        df_f, df_r, dr, _, _, _ = model.calculate_forces(
            speed=speed,
            aero_mode=mode,
            ride_height_front=ride_height_front,
            ride_height_rear=ride_height_rear,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )
        downforce_front.append(df_f)
        downforce_rear.append(df_r)
        drag.append(dr)
    return np.asarray(downforce_front), np.asarray(downforce_rear), np.asarray(drag)


def plot_mode_force_curves() -> plt.Figure:
    """Plot downforce and drag vs speed for both active-aero modes."""
    model = AerodynamicsModel()
    speeds = np.linspace(50.0, 360.0, 80)

    df_f_high, df_r_high, drag_high = _sweep_forces(model, speeds, "high_downforce")
    df_f_low, df_r_low, drag_low = _sweep_forces(model, speeds, "low_drag")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(speeds, df_f_high + df_r_high, lw=2.0, label="High Downforce")
    axes[0].plot(speeds, df_f_low + df_r_low, lw=2.0, label="Low Drag")
    axes[0].set_title("Total Downforce vs Speed")
    axes[0].set_xlabel("Speed [km/h]")
    axes[0].set_ylabel("Downforce [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(speeds, drag_high, lw=2.0, label="High Downforce")
    axes[1].plot(speeds, drag_low, lw=2.0, label="Low Drag")
    axes[1].set_title("Drag vs Speed")
    axes[1].set_xlabel("Speed [km/h]")
    axes[1].set_ylabel("Drag [N]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_ld_ratio() -> plt.Figure:
    """Plot aerodynamic efficiency L/D for both modes."""
    model = AerodynamicsModel()
    speeds = np.linspace(80.0, 360.0, 70)

    df_f_h, df_r_h, drag_h = _sweep_forces(model, speeds, "high_downforce")
    df_f_l, df_r_l, drag_l = _sweep_forces(model, speeds, "low_drag")

    ld_h = (df_f_h + df_r_h) / np.maximum(drag_h, 1.0)
    ld_l = (df_f_l + df_r_l) / np.maximum(drag_l, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(speeds, ld_h, lw=2.0, label="High Downforce")
    ax.plot(speeds, ld_l, lw=2.0, label="Low Drag")
    ax.set_title("Aero Efficiency (L/D)")
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("L/D [-]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ground_effect_sensitivity(speed_kmh: float = 250.0) -> plt.Figure:
    """Plot downforce sensitivity to ride height."""
    model = AerodynamicsModel()
    heights = np.linspace(0.02, 0.11, 80)
    total_downforce = []

    speed = speed_kmh / 3.6
    for rh in heights:
        df_f, df_r, _, _, _, _ = model.calculate_forces(
            speed=speed,
            aero_mode="high_downforce",
            ride_height_front=rh,
            ride_height_rear=rh + 0.005,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )
        total_downforce.append(df_f + df_r)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(heights * 1000.0, np.asarray(total_downforce), lw=2.0)
    ax.set_title("Ground Effect Sensitivity")
    ax.set_xlabel("Ride Height [mm]")
    ax.set_ylabel("Total Downforce [N]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dirty_air_effect(speed_kmh: float = 280.0) -> plt.Figure:
    """Plot downforce loss versus following distance."""
    model = AerodynamicsModel()
    distances = np.linspace(0.5, 20.0, 100)
    downforce = []
    speed = speed_kmh / 3.6

    for dist in distances:
        df_f, df_r, _, _, _, _ = model.calculate_forces(
            speed=speed,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            distance_to_car_ahead=dist,
        )
        downforce.append(df_f + df_r)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(distances, np.asarray(downforce), lw=2.0)
    ax.set_title("Dirty Air Impact")
    ax.set_xlabel("Distance to Car Ahead [m]")
    ax.set_ylabel("Total Downforce [N]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_aero_map() -> plt.Figure:
    """Render aero map of drag and downforce across speed and ride-height."""
    model = AerodynamicsModel()
    speeds = np.linspace(120.0, 340.0, 50)
    heights = np.linspace(0.03, 0.09, 40)

    total_downforce = np.zeros((len(heights), len(speeds)))
    total_drag = np.zeros_like(total_downforce)

    for i, rh in enumerate(heights):
        for j, speed_kmh in enumerate(speeds):
            df_f, df_r, drag, _, _, _ = model.calculate_forces(
                speed=speed_kmh / 3.6,
                aero_mode="high_downforce",
                ride_height_front=rh,
                ride_height_rear=rh + 0.005,
                pitch=0.0,
                roll=0.0,
                yaw=0.0,
            )
            total_downforce[i, j] = df_f + df_r
            total_drag[i, j] = drag

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(
        total_downforce,
        aspect="auto",
        origin="lower",
        extent=[speeds.min(), speeds.max(), heights.min() * 1000, heights.max() * 1000],
    )
    axes[0].set_title("Downforce Map [N]")
    axes[0].set_xlabel("Speed [km/h]")
    axes[0].set_ylabel("Ride Height [mm]")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        total_drag,
        aspect="auto",
        origin="lower",
        extent=[speeds.min(), speeds.max(), heights.min() * 1000, heights.max() * 1000],
    )
    axes[1].set_title("Drag Map [N]")
    axes[1].set_xlabel("Speed [km/h]")
    axes[1].set_ylabel("Ride Height [mm]")
    fig.colorbar(im1, ax=axes[1])

    return fig


def plot_mode_transition(duration_s: float = 0.5, dt: float = 0.01) -> plt.Figure:
    """Visualize smooth transition from high-downforce to low-drag mode."""
    model = AerodynamicsModel()
    model.set_aero_mode("low_drag")

    t = np.arange(0.0, duration_s, dt)
    df_trace = []
    drag_trace = []

    for _ in t:
        model.update(dt)
        df_f, df_r, drag, _, _, _ = model.calculate_forces(
            speed=250.0 / 3.6,
            aero_mode=model.target_mode,
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )
        df_trace.append(df_f + df_r)
        drag_trace.append(drag)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, df_trace, lw=2.0)
    axes[0].set_ylabel("Downforce [N]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, drag_trace, lw=2.0)
    axes[1].set_ylabel("Drag [N]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Active Aero Transition")
    fig.tight_layout()
    return fig
