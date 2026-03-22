#!/usr/bin/env python3
"""Demonstration script for Pacejka tire model behaviors."""

from __future__ import annotations

import numpy as np

from utils.tire_visualization import (
    plot_compound_comparison,
    plot_force_curves,
    plot_friction_ellipse,
    plot_temperature_history,
    plot_wear_progression,
)
from vehicle_dynamics.tire_model import TireModel


def demo_force_curves() -> None:
    tire = TireModel(compound="soft")
    fig1 = plot_force_curves(tire)
    fig1.savefig("logs/tire_force_curves.png", dpi=150)

    fig2 = plot_friction_ellipse(tire)
    fig2.savefig("logs/tire_friction_ellipse.png", dpi=150)


def demo_temperature_evolution() -> None:
    tire = TireModel(compound="soft")
    dt = 0.1
    sim_time = 120.0
    time_s = np.arange(0.0, sim_time, dt)

    inner, middle, outer, grip = [], [], [], []
    for _ in time_s:
        fx, fy, _ = tire.calculate_forces(
            slip_angle=np.deg2rad(6.0),
            slip_ratio=0.12,
            normal_load=4600.0,
            temperature=tire.state.surface_temp,
            wear=tire.state.wear_percentage,
            wheel_speed=240.0,
            vehicle_speed=70.0,
        )
        tire.update_temperature(
            forces={"Fx": fx, "Fy": fy},
            speeds={
                "vx_slip": 4.0,
                "vy_slip": 1.5,
                "track_temp": 45.0,
                "camber": np.deg2rad(-3.0),
            },
            ambient_temp=32.0,
            dt=dt,
        )
        tire.update_wear(np.deg2rad(6.0), 0.12, tire.state.surface_temp, dt)

        inner.append(tire.state.inner_temp)
        middle.append(tire.state.middle_temp)
        outer.append(tire.state.outer_temp)
        grip.append(
            float(
                tire.get_grip_multiplier(
                    tire.state.surface_temp, tire.state.wear_percentage
                )
            )
        )

    fig = plot_temperature_history(
        time_s,
        np.asarray(inner),
        np.asarray(middle),
        np.asarray(outer),
        np.asarray(grip),
    )
    fig.savefig("logs/tire_temperature_evolution.png", dpi=150)


def demo_stint_simulation() -> None:
    tire = TireModel(compound="medium")
    laps = np.arange(1, 31)
    wear = []
    grip = []
    life_left = []

    for _lap in laps:
        for _ in range(100):
            tire.update_wear(np.deg2rad(5.0), 0.08, tire.state.surface_temp, dt=0.2)
        wear.append(tire.state.wear_percentage)
        grip_now = float(
            tire.get_grip_multiplier(
                tire.state.surface_temp, tire.state.wear_percentage
            )
        )
        grip.append(grip_now)
        life_left.append(max(0.0, (100.0 - tire.state.wear_percentage) / 3.0))

    fig = plot_wear_progression(
        laps=laps,
        wear_pct=np.asarray(wear),
        grip_level=np.asarray(grip),
        remaining_life_laps=np.asarray(life_left),
    )
    fig.savefig("logs/tire_stint_progression.png", dpi=150)


def demo_compound_comparison() -> None:
    fig = plot_compound_comparison()
    fig.savefig("logs/tire_compound_comparison.png", dpi=150)


def main() -> None:
    demo_force_curves()
    demo_temperature_evolution()
    demo_stint_simulation()
    demo_compound_comparison()
    print("Tire model demo complete. Figures written to logs/.")


if __name__ == "__main__":
    main()
