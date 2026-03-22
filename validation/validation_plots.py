#!/usr/bin/env python3
"""Plotting utilities for validation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_tire_validation_curves(
    tire_model: Any, output_dir: str = "validation/plots"
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slip_deg = np.linspace(-18.0, 18.0, 200)
    slip_rad = np.deg2rad(slip_deg)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    for fz in [3000.0, 5000.0, 7000.0]:
        fy = tire_model.calculate_lateral_force(slip_rad, fz, 100.0, 0.0)
        ax1.plot(slip_deg, fy, label=f"Fz={int(fz)}N")
    ax1.set_title("Lateral Force vs Slip Angle")
    ax1.set_xlabel("Slip Angle [deg]")
    ax1.set_ylabel("Fy [N]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    p1 = out_dir / "tire_lateral_force_validation.png"
    _ensure_parent(p1)
    fig1.tight_layout()
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    temps = np.asarray([60.0, 80.0, 100.0, 120.0, 140.0])
    grip = []
    for t in temps:
        fy = abs(
            float(tire_model.calculate_lateral_force(np.deg2rad(8.0), 5000.0, t, 0.0))
        )
        grip.append(fy)
    grip = np.asarray(grip)
    grip /= max(np.max(grip), 1.0)
    ax2.plot(temps, grip, marker="o")
    ax2.set_title("Temperature Grip Sensitivity")
    ax2.set_xlabel("Temperature [C]")
    ax2.set_ylabel("Normalized Grip")
    ax2.grid(True, alpha=0.3)
    p2 = out_dir / "tire_temperature_validation.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)

    return [str(p1), str(p2)]


def plot_aero_validation(
    aero_model: Any, output_dir: str = "validation/plots"
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speeds = np.linspace(80.0, 340.0, 80)
    df_high, df_low, drag_high, drag_low = [], [], [], []

    for s in speeds:
        v = s / 3.6
        dff_h, dfr_h = aero_model.calculate_downforce(v, "high_downforce", 0.05, 0.055)
        dff_l, dfr_l = aero_model.calculate_downforce(v, "low_drag", 0.05, 0.055)
        df_high.append(dff_h + dfr_h)
        df_low.append(dff_l + dfr_l)
        drag_high.append(aero_model.calculate_total_drag(v, "high_downforce"))
        drag_low.append(aero_model.calculate_total_drag(v, "low_drag"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(speeds, df_high, label="High DF")
    axes[0].plot(speeds, df_low, label="Low Drag")
    axes[0].set_title("Downforce Validation")
    axes[0].set_xlabel("Speed [km/h]")
    axes[0].set_ylabel("Downforce [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(speeds, drag_high, label="High DF")
    axes[1].plot(speeds, drag_low, label="Low Drag")
    axes[1].set_title("Drag Validation")
    axes[1].set_xlabel("Speed [km/h]")
    axes[1].set_ylabel("Drag [N]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out = out_dir / "aero_validation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return [str(out)]


def plot_acceleration_curve(
    vehicle_model: Any, output_dir: str = "validation/plots"
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vehicle_model.reset(110.0)
    dt = 0.01
    t = np.arange(0.0, 12.0, dt)
    speed = []

    for _ in t:
        vehicle_model.update(
            dt, steering=0.0, throttle=1.0, brake=0.0, aero_mode="low_drag"
        )
        speed.append(vehicle_model.get_speed_kmh())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, speed, lw=2.0)
    ax.axhline(100.0, color="k", ls="--", alpha=0.4)
    ax.axhline(200.0, color="k", ls="--", alpha=0.4)
    ax.axhline(300.0, color="k", ls="--", alpha=0.4)
    ax.set_title("Acceleration Validation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.grid(True, alpha=0.3)
    out = out_dir / "vehicle_acceleration_validation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return [str(out)]


def plot_lap_simulation(
    speed_trace: np.ndarray, output_dir: str = "validation/plots"
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(speed_trace, lw=1.6)
    ax.set_title("Lap Simulation Speed Trace")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed [km/h]")
    ax.grid(True, alpha=0.3)
    out = out_dir / "lap_simulation_speed_trace.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return [str(out)]
