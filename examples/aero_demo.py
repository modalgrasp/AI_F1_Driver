#!/usr/bin/env python3
"""Demonstration script for active aerodynamics model."""

from __future__ import annotations

import numpy as np

from utils.aero_visualization import (
    plot_aero_map,
    plot_dirty_air_effect,
    plot_ground_effect_sensitivity,
    plot_ld_ratio,
    plot_mode_force_curves,
    plot_mode_transition,
)
from vehicle_dynamics.aerodynamics import AerodynamicsModel


def demo_force_curves() -> None:
    fig = plot_mode_force_curves()
    fig.savefig("logs/aero_mode_force_curves.png", dpi=150)


def demo_mode_comparison() -> None:
    fig = plot_ld_ratio()
    fig.savefig("logs/aero_ld_ratio.png", dpi=150)


def demo_lap_mode_switching() -> None:
    model = AerodynamicsModel()
    dt = 0.02
    steps = 500

    total_drag = []
    total_downforce = []
    speed_trace = np.linspace(120.0, 330.0, steps)

    for i in range(steps):
        speed = speed_trace[i] / 3.6
        mode = "low_drag" if speed_trace[i] > 250.0 else "high_downforce"
        model.set_aero_mode(mode)
        model.update(dt)

        df_f, df_r, drag, _, _, _ = model.calculate_forces(
            speed=speed,
            aero_mode=mode,
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=np.deg2rad(0.3 * np.sin(i * 0.03)),
            roll=np.deg2rad(0.6 * np.sin(i * 0.04)),
            yaw=np.deg2rad(0.4 * np.sin(i * 0.05)),
        )
        total_downforce.append(df_f + df_r)
        total_drag.append(drag)

    print(
        "Lap-mode switching summary:",
        {
            "avg_downforce_N": float(np.mean(total_downforce)),
            "avg_drag_N": float(np.mean(total_drag)),
            "max_downforce_N": float(np.max(total_downforce)),
            "max_drag_N": float(np.max(total_drag)),
        },
    )


def demo_dirty_air_impact() -> None:
    fig = plot_dirty_air_effect()
    fig.savefig("logs/aero_dirty_air.png", dpi=150)


def main() -> None:
    demo_force_curves()
    demo_mode_comparison()
    demo_lap_mode_switching()
    demo_dirty_air_impact()
    plot_ground_effect_sensitivity().savefig("logs/aero_ground_effect.png", dpi=150)
    plot_aero_map().savefig("logs/aero_map.png", dpi=150)
    plot_mode_transition().savefig("logs/aero_transition.png", dpi=150)
    print("Aerodynamics demo complete. Figures written to logs/.")


if __name__ == "__main__":
    main()
