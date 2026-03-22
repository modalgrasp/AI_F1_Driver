#!/usr/bin/env python3
"""Demo script for 2026 hybrid powertrain model."""

from __future__ import annotations

import numpy as np

from utils.powertrain_visualization import (
    plot_deployment_harvest_map,
    plot_energy_flows_lap,
    plot_power_curves,
    plot_torque_by_gear,
)
from vehicle_dynamics.energy_management import EnergyManagementSystem
from vehicle_dynamics.powertrain import PowertrainModel


def demo_power_curves() -> None:
    plot_power_curves().savefig("logs/powertrain_power_curves.png", dpi=150)
    plot_torque_by_gear().savefig("logs/powertrain_torque_by_gear.png", dpi=150)


def demo_lap_energy_simulation() -> None:
    plot_energy_flows_lap().savefig("logs/powertrain_lap_energy.png", dpi=150)


def demo_strategy_comparison() -> None:
    model = PowertrainModel()
    ems = EnergyManagementSystem({"track_length": 5500.0})

    dt = 0.02
    t = np.arange(0.0, 60.0, dt)
    totals: dict[str, list[float]] = {
        "max_power": [],
        "balanced": [],
        "save_energy": [],
    }

    for strategy in totals:
        model.reset(110.0)
        for ti in t:
            model.update(dt)
            speed = 70.0 + 15.0 * np.sin(ti * 0.15)
            rpm = model.transmission.calculate_engine_rpm(
                speed / model.wheel_radius, model.transmission.current_gear
            )
            rpm = float(np.clip(rpm, model.ice.idle_rpm, model.ice.max_rpm))
            brake = max(0.0, np.sin(ti * 0.31 - 0.8))
            throttle = np.clip(0.85 - 0.75 * brake, 0.0, 1.0)

            harvest = ems.get_harvest_strategy((ti / 60.0) % 1.0, model.battery.soc)
            _, _, _, state = model.calculate_wheel_power(
                throttle=throttle,
                brake=brake,
                rpm=rpm,
                gear=model.transmission.current_gear,
                vehicle_speed=speed,
                deployment_strategy=strategy,
                harvest_strategy=harvest,
                dt=dt,
            )
            totals[strategy].append(state["total_power_w"] / 1000.0)

        print(
            f"{strategy}: avg power = {float(np.mean(totals[strategy])):.1f} kW, "
            f"final SOC = {model.battery.soc:.3f}, fuel left = {model.fuel_system.fuel_remaining:.2f} kg"
        )

    plot_deployment_harvest_map().savefig("logs/powertrain_strategy_map.png", dpi=150)


def demo_overtake_mode() -> None:
    model = PowertrainModel()
    model.battery.state.state_of_charge = 0.7
    model.battery.state.energy_j = 0.7 * model.battery.capacity

    dt = 0.02
    for i in range(400):
        model.update(dt)
        speed = 80.0
        rpm = model.transmission.calculate_engine_rpm(speed / model.wheel_radius, 7)
        gap = 0.8 if 100 < i < 220 else 1.5
        _, _, _, state = model.calculate_wheel_power(
            throttle=1.0,
            brake=0.0,
            rpm=rpm,
            gear=7,
            vehicle_speed=speed,
            deployment_strategy="max_power",
            harvest_strategy="balanced",
            gap_to_leader=gap,
            dt=dt,
        )
        if i % 50 == 0:
            print(
                {
                    "step": i,
                    "total_power_kw": round(state["total_power_w"] / 1000.0, 1),
                    "overtake_active": state["overtake_active"],
                    "soc": round(state["battery_soc"], 3),
                }
            )


def main() -> None:
    demo_power_curves()
    demo_lap_energy_simulation()
    demo_strategy_comparison()
    demo_overtake_mode()
    print("Powertrain demo complete. Plots saved under logs/.")


if __name__ == "__main__":
    main()
