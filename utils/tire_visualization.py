#!/usr/bin/env python3
"""Visualization helpers for tire force, temperature, and wear behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from vehicle_dynamics.tire_model import TireModel


@dataclass
class TireSweepResult:
    x: np.ndarray
    y: np.ndarray
    label: str


def lateral_force_curve(
    tire: TireModel,
    normal_load: float,
    temperature: float,
    wear: float,
    slip_deg_range: tuple[float, float] = (-15.0, 15.0),
    points: int = 200,
) -> TireSweepResult:
    """Generate Fy vs slip-angle curve."""
    slip_deg = np.linspace(slip_deg_range[0], slip_deg_range[1], points)
    slip_rad = np.deg2rad(slip_deg)
    fy = tire.calculate_lateral_force(slip_rad, normal_load, temperature, wear)
    return TireSweepResult(slip_deg, np.asarray(fy), "Fy vs Slip Angle")


def longitudinal_force_curve(
    tire: TireModel,
    normal_load: float,
    temperature: float,
    wear: float,
    slip_ratio_range: tuple[float, float] = (-0.3, 0.3),
    points: int = 220,
) -> TireSweepResult:
    """Generate Fx vs slip-ratio curve."""
    slip_ratio = np.linspace(slip_ratio_range[0], slip_ratio_range[1], points)
    fx = tire.calculate_longitudinal_force(slip_ratio, normal_load, temperature, wear)
    return TireSweepResult(slip_ratio, np.asarray(fx), "Fx vs Slip Ratio")


def friction_ellipse(
    tire: TireModel,
    normal_load: float,
    temperature: float,
    wear: float,
    slip_angle_deg: Iterable[float] = tuple(np.linspace(-12.0, 12.0, 50)),
    slip_ratio_values: Iterable[float] = tuple(np.linspace(-0.2, 0.2, 50)),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fx/Fy points under combined-slip requests."""
    fx_points: list[float] = []
    fy_points: list[float] = []
    for alpha_deg in slip_angle_deg:
        for kappa in slip_ratio_values:
            fx, fy = tire.calculate_combined_forces(
                np.deg2rad(alpha_deg),
                kappa,
                normal_load,
                temperature,
                wear,
            )
            fx_points.append(float(fx))
            fy_points.append(float(fy))
    return np.asarray(fx_points), np.asarray(fy_points)


def plot_force_curves(tire: TireModel, normal_load: float = 4500.0) -> plt.Figure:
    """Plot lateral and longitudinal force sweeps for one tire."""
    lat = lateral_force_curve(tire, normal_load=normal_load, temperature=95.0, wear=5.0)
    lon = longitudinal_force_curve(
        tire, normal_load=normal_load, temperature=95.0, wear=5.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(lat.x, lat.y, lw=2.0)
    axes[0].set_title("Lateral Force Curve")
    axes[0].set_xlabel("Slip Angle [deg]")
    axes[0].set_ylabel("Fy [N]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lon.x, lon.y, lw=2.0)
    axes[1].set_title("Longitudinal Force Curve")
    axes[1].set_xlabel("Slip Ratio [-]")
    axes[1].set_ylabel("Fx [N]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_friction_ellipse(tire: TireModel, normal_load: float = 4500.0) -> plt.Figure:
    """Plot friction-ellipse cloud for combined slip."""
    fx, fy = friction_ellipse(tire, normal_load, temperature=95.0, wear=8.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fx, fy, s=5, alpha=0.35)
    ax.set_title("Combined Slip Friction Envelope")
    ax.set_xlabel("Fx [N]")
    ax.set_ylabel("Fy [N]")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_temperature_history(
    time_s: np.ndarray,
    inner: np.ndarray,
    middle: np.ndarray,
    outer: np.ndarray,
    grip_multiplier: np.ndarray,
) -> plt.Figure:
    """Plot 3-zone temperatures and associated grip multiplier."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_s, inner, label="Inner", lw=1.8)
    axes[0].plot(time_s, middle, label="Middle", lw=1.8)
    axes[0].plot(time_s, outer, label="Outer", lw=1.8)
    axes[0].set_ylabel("Temperature [C]")
    axes[0].set_title("Tire Temperature Zones")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, grip_multiplier, color="tab:green", lw=2.0)
    axes[1].set_ylabel("Grip Multiplier [-]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_wear_progression(
    laps: np.ndarray,
    wear_pct: np.ndarray,
    grip_level: np.ndarray,
    remaining_life_laps: np.ndarray,
) -> plt.Figure:
    """Plot wear evolution and remaining life estimate."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(laps, wear_pct, lw=2.0, color="tab:red")
    axes[0].set_ylabel("Wear [%]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(laps, grip_level, lw=2.0, color="tab:blue")
    axes[1].set_ylabel("Grip Multiplier")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(laps, remaining_life_laps, lw=2.0, color="tab:purple")
    axes[2].set_ylabel("Remaining Life [laps]")
    axes[2].set_xlabel("Lap")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_compound_comparison(
    compounds: list[str] = ["soft", "medium", "hard"],
    normal_load: float = 4500.0,
) -> plt.Figure:
    """Compare lateral force and wear behavior across compounds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    slip_deg = np.linspace(-12.0, 12.0, 200)
    slip_rad = np.deg2rad(slip_deg)

    for compound in compounds:
        tire = TireModel(compound=compound)
        fy = tire.calculate_lateral_force(slip_rad, normal_load, 95.0, 5.0)
        axes[0].plot(slip_deg, fy, lw=2.0, label=compound.capitalize())

        wear = np.linspace(0.0, 100.0, 100)
        grip = np.asarray(tire.get_grip_multiplier(95.0, wear))
        axes[1].plot(wear, grip, lw=2.0, label=compound.capitalize())

    axes[0].set_title("Lateral Force by Compound")
    axes[0].set_xlabel("Slip Angle [deg]")
    axes[0].set_ylabel("Fy [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Grip vs Wear by Compound")
    axes[1].set_xlabel("Wear [%]")
    axes[1].set_ylabel("Grip Multiplier [-]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig
