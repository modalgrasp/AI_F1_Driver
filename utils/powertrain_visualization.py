#!/usr/bin/env python3
"""Visualization tools for 2026 hybrid powertrain behavior."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from vehicle_dynamics.energy_management import EnergyManagementSystem
from vehicle_dynamics.powertrain import PowertrainModel


def plot_power_curves() -> plt.Figure:
    """Plot ICE, MGU-K, and combined power vs RPM."""
    model = PowertrainModel()
    rpm = np.linspace(5000.0, 15000.0, 220)
    ice_power = []
    electric_power = []
    total_power = []

    for r in rpm:
        model.update(0.01)
        _, _, _, state = model.calculate_wheel_power(
            throttle=1.0,
            brake=0.0,
            rpm=float(r),
            gear=6,
            vehicle_speed=70.0,
            deployment_strategy="balanced",
            dt=0.01,
        )
        ice_power.append(state["ice_power_w"] / 1000.0)
        electric_power.append(state["mguk_power_w"] / 1000.0)
        total_power.append(state["total_power_w"] / 1000.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rpm, ice_power, lw=2.0, label="ICE")
    ax.plot(rpm, electric_power, lw=2.0, label="MGU-K")
    ax.plot(rpm, total_power, lw=2.4, label="Combined")
    ax.set_title("Power Curves")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Power [kW]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_torque_by_gear(
    vehicle_speed_range: tuple[float, float] = (20.0, 100.0)
) -> plt.Figure:
    """Plot wheel torque across gears over vehicle speed."""
    model = PowertrainModel()
    speeds = np.linspace(vehicle_speed_range[0], vehicle_speed_range[1], 120)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for gear in range(1, 9):
        torque = []
        for v in speeds:
            wheel_speed = v / model.wheel_radius
            rpm = model.transmission.calculate_engine_rpm(wheel_speed, gear)
            rpm = float(np.clip(rpm, model.ice.idle_rpm, model.ice.max_rpm))
            model.update(0.01)
            wheel_torque, _, _, _ = model.calculate_wheel_power(
                throttle=1.0,
                brake=0.0,
                rpm=rpm,
                gear=gear,
                vehicle_speed=v,
                deployment_strategy="balanced",
                dt=0.01,
            )
            torque.append(wheel_torque)
        ax.plot(speeds * 3.6, torque, lw=1.6, label=f"Gear {gear}")

    ax.set_title("Wheel Torque by Gear")
    ax.set_xlabel("Vehicle Speed [km/h]")
    ax.set_ylabel("Wheel Torque [Nm]")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_energy_flows_lap(duration_s: float = 95.0, dt: float = 0.05) -> plt.Figure:
    """Simulate a synthetic lap and plot SOC/fuel/power flows."""
    model = PowertrainModel()
    ems = EnergyManagementSystem({"track_length": 5500.0})

    t = np.arange(0.0, duration_s, dt)
    soc = []
    fuel = []
    power = []
    track_pos = []

    for ti in t:
        model.update(dt)
        speed = 55.0 + 25.0 * np.sin(ti * 0.09)
        braking = max(0.0, np.sin(ti * 0.35 - 1.2))
        throttle = np.clip(0.9 - 0.8 * braking, 0.0, 1.0)
        position = (ti / duration_s) % 1.0
        actions = ems.get_control_actions(position, model.battery.soc, lap_type="race")

        rpm = model.transmission.calculate_engine_rpm(
            speed / model.wheel_radius, model.transmission.current_gear
        )
        rpm = float(np.clip(rpm, model.ice.idle_rpm, model.ice.max_rpm))

        _, _, _, state = model.calculate_wheel_power(
            throttle=throttle,
            brake=braking,
            rpm=rpm,
            gear=model.transmission.current_gear,
            vehicle_speed=speed,
            deployment_strategy=actions["deployment_strategy"],
            harvest_strategy=actions["harvest_strategy"],
            dt=dt,
        )

        soc.append(state["battery_soc"])
        fuel.append(state["fuel_remaining_kg"])
        power.append(state["total_power_w"] / 1000.0)
        track_pos.append(position)

        if rpm > 14000 and model.transmission.current_gear < 8:
            model.shift_gear(+1)
        elif rpm < 6500 and model.transmission.current_gear > 1:
            model.shift_gear(-1)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
    axes[0].plot(t, power, lw=1.8)
    axes[0].set_ylabel("Power [kW]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, soc, lw=1.8, color="tab:green")
    axes[1].set_ylabel("SOC [-]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, fuel, lw=1.8, color="tab:red")
    axes[2].set_ylabel("Fuel [kg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Lap Energy and Fuel Evolution")
    fig.tight_layout()
    return fig


def plot_deployment_harvest_map() -> plt.Figure:
    """Plot strategy map over normalized lap position."""
    ems = EnergyManagementSystem({"track_length": 5500.0})
    x = np.linspace(0.0, 1.0, 300)

    deploy = []
    harvest = []
    for pos in x:
        d = ems.get_deployment_strategy(pos, battery_soc=0.55)
        h = ems.get_harvest_strategy(pos, battery_soc=0.55)
        deploy.append({"save_energy": 0, "balanced": 1, "max_power": 2}[d])
        harvest.append({"conservative": 0, "balanced": 1, "aggressive": 2}[h])

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 4.8), sharex=True)
    axes[0].plot(x, deploy, lw=2.0)
    axes[0].set_yticks([0, 1, 2], ["save", "balanced", "max"])
    axes[0].set_ylabel("Deploy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, harvest, lw=2.0)
    axes[1].set_yticks([0, 1, 2], ["consv", "balanced", "aggr"])
    axes[1].set_ylabel("Harvest")
    axes[1].set_xlabel("Normalized Lap Position")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
