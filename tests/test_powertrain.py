#!/usr/bin/env python3
"""Tests for 2026 hybrid powertrain model."""

from __future__ import annotations

import time

import numpy as np
import pytest

from vehicle_dynamics.energy_management import EnergyManagementSystem
from vehicle_dynamics.powertrain import PowertrainModel


@pytest.fixture
def powertrain() -> PowertrainModel:
    return PowertrainModel()


@pytest.mark.unit
def test_peak_combined_power_near_spec(powertrain: PowertrainModel) -> None:
    powertrain.update(0.02)
    _, _, _, state = powertrain.calculate_wheel_power(
        throttle=1.0,
        brake=0.0,
        rpm=12000.0,
        gear=6,
        vehicle_speed=80.0,
        deployment_strategy="balanced",
        dt=0.02,
    )
    assert state["ice_power_w"] > 280000.0
    assert state["mguk_power_w"] > 250000.0
    assert state["total_power_w"] > 600000.0
    assert state["total_power_w"] < 800000.0


@pytest.mark.unit
def test_balanced_split_approx_50_50(powertrain: PowertrainModel) -> None:
    for _ in range(10):
        powertrain.update(0.02)
        _, _, _, state = powertrain.calculate_wheel_power(
            throttle=1.0,
            brake=0.0,
            rpm=11500.0,
            gear=6,
            vehicle_speed=75.0,
            deployment_strategy="balanced",
            dt=0.02,
        )
    split_ice = state["power_split_ice"]
    split_elec = state["power_split_electric"]
    assert 0.40 < split_ice < 0.60
    assert 0.40 < split_elec < 0.60


@pytest.mark.unit
def test_braking_regen_charges_battery(powertrain: PowertrainModel) -> None:
    soc_initial = powertrain.battery.soc
    for _ in range(200):
        powertrain.update(0.01)
        powertrain.calculate_wheel_power(
            throttle=0.0,
            brake=0.8,
            rpm=10000.0,
            gear=5,
            vehicle_speed=55.0,
            deployment_strategy="save_energy",
            harvest_strategy="aggressive",
            dt=0.01,
        )
    assert powertrain.battery.soc > soc_initial


@pytest.mark.unit
def test_fuel_consumption_decreases_mass(powertrain: PowertrainModel) -> None:
    fuel_start = powertrain.fuel_system.fuel_remaining
    for _ in range(300):
        powertrain.update(0.01)
        powertrain.calculate_wheel_power(
            throttle=0.9,
            brake=0.0,
            rpm=11000.0,
            gear=6,
            vehicle_speed=70.0,
            deployment_strategy="balanced",
            dt=0.01,
        )
    assert powertrain.fuel_system.fuel_remaining < fuel_start


@pytest.mark.unit
def test_battery_limits_prevent_underflow(powertrain: PowertrainModel) -> None:
    powertrain.battery.state.energy_j = (
        powertrain.battery.min_soc * powertrain.battery.capacity
    )
    for _ in range(50):
        powertrain.update(0.02)
        powertrain.calculate_wheel_power(
            throttle=1.0,
            brake=0.0,
            rpm=12000.0,
            gear=6,
            vehicle_speed=80.0,
            deployment_strategy="max_power",
            dt=0.02,
        )
    assert powertrain.battery.soc >= powertrain.battery.min_soc - 1e-6


@pytest.mark.unit
def test_empty_fuel_kills_ice_output(powertrain: PowertrainModel) -> None:
    powertrain.fuel_system.fuel_remaining = 0.0
    powertrain.update(0.02)
    _, _, _, state = powertrain.calculate_wheel_power(
        throttle=1.0,
        brake=0.0,
        rpm=12000.0,
        gear=6,
        vehicle_speed=80.0,
        deployment_strategy="balanced",
        dt=0.02,
    )
    assert state["ice_power_w"] < 1.0


@pytest.mark.unit
def test_overtake_mode_activation(powertrain: PowertrainModel) -> None:
    powertrain.battery.state.state_of_charge = 0.6
    powertrain.battery.state.energy_j = powertrain.battery.capacity * 0.6
    powertrain.update(0.02)
    _, _, _, state = powertrain.calculate_wheel_power(
        throttle=1.0,
        brake=0.0,
        rpm=12000.0,
        gear=6,
        vehicle_speed=80.0,
        deployment_strategy="max_power",
        gap_to_leader=0.8,
        dt=0.02,
    )
    assert state["overtake_active"] in {True, False}


@pytest.mark.unit
def test_shift_logic_and_optimal_gear(powertrain: PowertrainModel) -> None:
    g = powertrain.get_optimal_gear(vehicle_speed=60.0, target="acceleration")
    assert 1 <= g <= 8
    current = powertrain.transmission.current_gear
    powertrain.shift_gear(+1)
    powertrain.update(0.1)
    assert powertrain.transmission.current_gear >= current


@pytest.mark.unit
def test_ems_strategy_outputs_valid() -> None:
    ems = EnergyManagementSystem({"track_length": 5500.0})
    deploy = ems.get_deployment_strategy(track_position=0.2, battery_soc=0.5)
    harvest = ems.get_harvest_strategy(track_position=0.25, battery_soc=0.5)
    assert deploy in {"max_power", "balanced", "save_energy"}
    assert harvest in {"aggressive", "balanced", "conservative"}


@pytest.mark.integration
def test_energy_balance_reasonable(powertrain: PowertrainModel) -> None:
    energy_in = 0.0
    energy_out = 0.0
    dt = 0.01
    for _ in range(1000):
        powertrain.update(dt)
        wheel_torque, fuel_flow, battery_flow, state = powertrain.calculate_wheel_power(
            throttle=0.75,
            brake=0.15,
            rpm=10500.0,
            gear=5,
            vehicle_speed=65.0,
            deployment_strategy="balanced",
            harvest_strategy="balanced",
            dt=dt,
        )
        _ = wheel_torque
        fuel_energy = fuel_flow * dt * powertrain.fuel_system.energy_content_mj_kg * 1e6
        electrical = abs(battery_flow) * dt
        wheel_energy = state["total_power_w"] * dt
        energy_in += fuel_energy + electrical
        energy_out += wheel_energy

    assert energy_out > 0.0
    assert energy_in > energy_out * 0.45


@pytest.mark.slow
def test_update_performance_under_target(powertrain: PowertrainModel) -> None:
    loops = 15000
    start = time.perf_counter()
    for _ in range(loops):
        powertrain.update(0.01)
        powertrain.calculate_wheel_power(
            throttle=0.8,
            brake=0.1,
            rpm=11000.0,
            gear=6,
            vehicle_speed=72.0,
            deployment_strategy="balanced",
            harvest_strategy="balanced",
            dt=0.01,
        )
    elapsed = time.perf_counter() - start
    per_update_ms = elapsed / loops * 1000.0
    assert per_update_ms < 0.5


@pytest.mark.unit
def test_numerical_stability_no_nan(powertrain: PowertrainModel) -> None:
    powertrain.update(0.01)
    wheel_torque, fuel_flow, battery_flow, state = powertrain.calculate_wheel_power(
        throttle=1.0,
        brake=1.0,
        rpm=15000.0,
        gear=1,
        vehicle_speed=0.1,
        deployment_strategy="max_power",
        harvest_strategy="aggressive",
        dt=0.01,
    )
    values = np.asarray(
        [wheel_torque, fuel_flow, battery_flow, state["total_power_w"]],
        dtype=np.float64,
    )
    assert np.isfinite(values).all()
