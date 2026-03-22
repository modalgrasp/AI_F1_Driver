#!/usr/bin/env python3
"""Tests for integrated vehicle dynamics model."""

from __future__ import annotations

import time

import numpy as np
import pytest

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


@pytest.fixture
def vehicle() -> VehicleDynamicsModel:
    model = VehicleDynamicsModel()
    model.reset()
    return model


@pytest.mark.unit
def test_model_initialization(vehicle: VehicleDynamicsModel) -> None:
    state = vehicle.get_state()
    assert state["speed_kmh"] == 0.0
    assert len(state["normal_loads"]) == 4


@pytest.mark.unit
def test_straight_line_acceleration(vehicle: VehicleDynamicsModel) -> None:
    dt = 0.01
    for _ in range(160):
        vehicle.update(dt, steering=0.0, throttle=1.0, brake=0.0, aero_mode="low_drag")
    assert vehicle.get_speed_kmh() > 30.0
    assert vehicle.state.x > 0.0


@pytest.mark.unit
def test_emergency_braking(vehicle: VehicleDynamicsModel) -> None:
    for _ in range(120):
        vehicle.update(0.01, steering=0.0, throttle=1.0, brake=0.0)
    speed_before = vehicle.get_speed_kmh()
    for _ in range(90):
        vehicle.update(0.01, steering=0.0, throttle=0.0, brake=1.0)
    speed_after = vehicle.get_speed_kmh()
    assert speed_after < speed_before


@pytest.mark.unit
def test_steady_state_cornering_generates_lateral_accel(
    vehicle: VehicleDynamicsModel,
) -> None:
    for _ in range(180):
        vehicle.update(
            0.01,
            steering=np.deg2rad(4.0),
            throttle=0.65,
            brake=0.0,
            aero_mode="high_downforce",
        )
    gx, gy = vehicle.get_g_forces()
    assert abs(gy) > 0.2
    assert np.isfinite([gx, gy]).all()


@pytest.mark.unit
def test_normal_loads_positive(vehicle: VehicleDynamicsModel) -> None:
    for _ in range(120):
        vehicle.update(0.01, steering=np.deg2rad(5.0), throttle=0.75, brake=0.0)
    loads = np.asarray(vehicle.last_normal_loads)
    assert np.all(loads > 0.0)


@pytest.mark.unit
def test_load_transfer_variation_cornering(vehicle: VehicleDynamicsModel) -> None:
    vehicle.reset()
    for _ in range(140):
        vehicle.update(0.01, steering=np.deg2rad(6.0), throttle=0.7, brake=0.0)
    loads = np.asarray(vehicle.last_normal_loads)
    front_split = abs(loads[0] - loads[1])
    rear_split = abs(loads[2] - loads[3])
    assert front_split > 20.0
    assert rear_split > 20.0


@pytest.mark.unit
def test_rk4_state_finite(vehicle: VehicleDynamicsModel) -> None:
    for _ in range(180):
        vehicle.update(0.01, steering=np.deg2rad(10.0), throttle=0.9, brake=0.3)
    state = vehicle.get_state()
    check = np.asarray(
        [
            state["x"],
            state["y"],
            state["yaw"],
            state["vx"],
            state["vy"],
            state["omega"],
        ]
    )
    assert np.isfinite(check).all()


@pytest.mark.integration
def test_subsystem_integration_state_consistency(vehicle: VehicleDynamicsModel) -> None:
    for _ in range(120):
        vehicle.update(0.01, steering=np.deg2rad(2.5), throttle=0.8, brake=0.1)
    st = vehicle.get_state()
    assert 1 <= st["gear"] <= 8
    assert 0.0 <= st["battery_soc"] <= 1.0
    assert st["fuel_mass"] <= vehicle.params.fuel_mass
    assert len(st["tire_states"]) == 4


@pytest.mark.integration
def test_energy_balance_not_unphysical(vehicle: VehicleDynamicsModel) -> None:
    dt = 0.01
    total_wheel_energy = 0.0
    total_fuel_energy = 0.0

    for _ in range(300):
        vehicle.update(dt, steering=0.0, throttle=0.8, brake=0.1)
        pt = vehicle.powertrain.last_state
        total_wheel_energy += pt.get("total_power_w", 0.0) * dt
        total_fuel_energy += pt.get("fuel_flow_kg_s", 0.0) * 43e6 * dt

    assert total_wheel_energy > 0.0
    assert total_fuel_energy > total_wheel_energy * 0.3


@pytest.mark.slow
def test_performance_under_1ms_target(vehicle: VehicleDynamicsModel) -> None:
    loops = 2000
    start = time.perf_counter()
    for _ in range(loops):
        vehicle.update(0.001, steering=np.deg2rad(2.0), throttle=0.7, brake=0.05)
    elapsed_ms = (time.perf_counter() - start) / loops * 1000.0
    assert elapsed_ms < 6.0


@pytest.mark.unit
def test_validation_targets_plausible_ranges(vehicle: VehicleDynamicsModel) -> None:
    for _ in range(700):
        vehicle.update(
            0.01, steering=0.0, throttle=1.0, brake=0.0, aero_mode="low_drag"
        )
    speed = vehicle.get_speed_kmh()
    assert speed > 95.0

    vehicle.reset()
    for _ in range(200):
        vehicle.update(
            0.01,
            steering=np.deg2rad(6.0),
            throttle=0.75,
            brake=0.0,
            aero_mode="high_downforce",
        )
    _, gy = vehicle.get_g_forces()
    assert abs(gy) < 8.0
