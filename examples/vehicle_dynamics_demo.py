#!/usr/bin/env python3
"""Demonstrations for integrated vehicle dynamics model."""

from __future__ import annotations

import numpy as np

from utils.vehicle_visualization import (
    plot_gg_diagram,
    plot_load_transfer,
    plot_speed_trace,
    plot_tire_forces,
    plot_trajectory,
)
from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


def _run_scenario(
    model: VehicleDynamicsModel,
    duration: float,
    dt: float,
    steering_fn,
    throttle_fn,
    brake_fn,
    aero_mode: str,
) -> dict[str, np.ndarray]:
    t = np.arange(0.0, duration, dt)
    x_hist, y_hist, v_hist = [], [], []
    ax_g_hist, ay_g_hist = [], []
    tire_hist = []
    load_hist = []

    for ti in t:
        st = model.update(
            dt=dt,
            steering=float(steering_fn(ti)),
            throttle=float(throttle_fn(ti)),
            brake=float(brake_fn(ti)),
            aero_mode=aero_mode,
        )
        x_hist.append(st.x)
        y_hist.append(st.y)
        v_hist.append(model.get_speed_kmh())
        gx, gy = model.get_g_forces()
        ax_g_hist.append(gx)
        ay_g_hist.append(gy)

        tire_hist.append(model.last_tire_forces[:, :2].copy())
        load_hist.append(model.last_normal_loads.copy())

    return {
        "t": t,
        "x": np.asarray(x_hist),
        "y": np.asarray(y_hist),
        "speed": np.asarray(v_hist),
        "gx": np.asarray(ax_g_hist),
        "gy": np.asarray(ay_g_hist),
        "tire_forces": np.asarray(tire_hist),
        "loads": np.asarray(load_hist),
    }


def demo_straight_line_acceleration() -> None:
    model = VehicleDynamicsModel()
    model.reset()

    out = _run_scenario(
        model=model,
        duration=8.0,
        dt=0.01,
        steering_fn=lambda _t: 0.0,
        throttle_fn=lambda _t: 1.0,
        brake_fn=lambda _t: 0.0,
        aero_mode="low_drag",
    )

    plot_speed_trace(out["t"], out["speed"]).savefig(
        "logs/vehicle_speed_accel.png", dpi=150
    )
    plot_trajectory(out["x"], out["y"]).savefig("logs/vehicle_traj_accel.png", dpi=150)
    print({"scenario": "acceleration", "v_end_kmh": float(out["speed"][-1])})


def demo_steady_state_cornering() -> None:
    model = VehicleDynamicsModel()
    model.reset()

    out = _run_scenario(
        model=model,
        duration=10.0,
        dt=0.01,
        steering_fn=lambda _t: np.deg2rad(4.5),
        throttle_fn=lambda _t: 0.7,
        brake_fn=lambda _t: 0.0,
        aero_mode="high_downforce",
    )

    plot_trajectory(out["x"], out["y"]).savefig(
        "logs/vehicle_traj_cornering.png", dpi=150
    )
    plot_gg_diagram(out["gx"], out["gy"]).savefig(
        "logs/vehicle_gg_cornering.png", dpi=150
    )
    print(
        {
            "scenario": "cornering",
            "lat_g_peak": float(np.max(np.abs(out["gy"]))),
            "speed_mean_kmh": float(np.mean(out["speed"])),
        }
    )


def demo_emergency_braking() -> None:
    model = VehicleDynamicsModel()
    model.reset()

    out_accel = _run_scenario(
        model=model,
        duration=4.0,
        dt=0.01,
        steering_fn=lambda _t: 0.0,
        throttle_fn=lambda _t: 1.0,
        brake_fn=lambda _t: 0.0,
        aero_mode="low_drag",
    )
    v_before = float(out_accel["speed"][-1])

    out_brake = _run_scenario(
        model=model,
        duration=3.0,
        dt=0.01,
        steering_fn=lambda _t: 0.0,
        throttle_fn=lambda _t: 0.0,
        brake_fn=lambda _t: 1.0,
        aero_mode="high_downforce",
    )

    plot_speed_trace(out_brake["t"], out_brake["speed"]).savefig(
        "logs/vehicle_speed_braking.png", dpi=150
    )
    plot_load_transfer(out_brake["t"], out_brake["loads"]).savefig(
        "logs/vehicle_load_braking.png", dpi=150
    )
    print(
        {
            "scenario": "braking",
            "v_before_kmh": v_before,
            "v_after_kmh": float(out_brake["speed"][-1]),
            "max_long_g": float(np.max(np.abs(out_brake["gx"]))),
        }
    )


def demo_full_lap_simulation() -> None:
    model = VehicleDynamicsModel()
    model.reset()

    duration = 90.0
    dt = 0.01
    out = _run_scenario(
        model=model,
        duration=duration,
        dt=dt,
        steering_fn=lambda t: np.deg2rad(
            3.0 * np.sin(0.08 * t) + 4.0 * np.sin(0.21 * t)
        ),
        throttle_fn=lambda t: np.clip(
            0.85 - 0.65 * max(0.0, np.sin(0.19 * t - 0.8)), 0.0, 1.0
        ),
        brake_fn=lambda t: max(0.0, np.sin(0.19 * t - 0.8)),
        aero_mode="high_downforce",
    )

    plot_trajectory(out["x"], out["y"]).savefig("logs/vehicle_traj_lap.png", dpi=150)
    plot_gg_diagram(out["gx"], out["gy"]).savefig("logs/vehicle_gg_lap.png", dpi=150)
    plot_tire_forces(out["t"], out["tire_forces"]).savefig(
        "logs/vehicle_tire_forces_lap.png", dpi=150
    )
    plot_speed_trace(out["t"], out["speed"]).savefig(
        "logs/vehicle_speed_lap.png", dpi=150
    )

    print(
        {
            "scenario": "full_lap",
            "mean_speed_kmh": float(np.mean(out["speed"])),
            "max_speed_kmh": float(np.max(out["speed"])),
            "peak_lat_g": float(np.max(np.abs(out["gy"]))),
            "fuel_remaining_kg": float(model.state.fuel_mass),
            "battery_soc": float(model.state.battery_soc),
        }
    )


def main() -> None:
    demo_straight_line_acceleration()
    demo_steady_state_cornering()
    demo_emergency_braking()
    demo_full_lap_simulation()
    print("Vehicle dynamics demos complete. Outputs saved under logs/.")


if __name__ == "__main__":
    main()
