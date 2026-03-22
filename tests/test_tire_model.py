#!/usr/bin/env python3
"""Tests for Pacejka tire model implementation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from vehicle_dynamics.tire_model import TireModel


@pytest.fixture
def tire_medium() -> TireModel:
    return TireModel(compound="medium")


@pytest.mark.unit
def test_magic_formula_lateral_reasonable_peak(tire_medium: TireModel) -> None:
    slip = np.deg2rad(np.linspace(-15.0, 15.0, 200))
    fy = np.asarray(tire_medium.calculate_lateral_force(slip, 4500.0, 95.0, 0.0))

    assert np.max(np.abs(fy)) > 8000.0
    assert np.max(np.abs(fy)) < 30000.0


@pytest.mark.unit
def test_magic_formula_longitudinal_reasonable_peak(tire_medium: TireModel) -> None:
    kappa = np.linspace(-0.3, 0.3, 200)
    fx = np.asarray(tire_medium.calculate_longitudinal_force(kappa, 4500.0, 95.0, 0.0))

    assert np.max(np.abs(fx)) > 8000.0
    assert np.max(np.abs(fx)) < 30000.0


@pytest.mark.unit
def test_load_sensitivity_increases_force_with_load(tire_medium: TireModel) -> None:
    fy_low = abs(
        float(tire_medium.calculate_lateral_force(np.deg2rad(7.0), 3000.0, 95.0, 0.0))
    )
    fy_high = abs(
        float(tire_medium.calculate_lateral_force(np.deg2rad(7.0), 6000.0, 95.0, 0.0))
    )
    assert fy_high > fy_low


@pytest.mark.unit
def test_wear_reduces_force(tire_medium: TireModel) -> None:
    fy_new = abs(
        float(tire_medium.calculate_lateral_force(np.deg2rad(6.0), 4500.0, 95.0, 0.0))
    )
    fy_old = abs(
        float(tire_medium.calculate_lateral_force(np.deg2rad(6.0), 4500.0, 95.0, 90.0))
    )
    assert fy_old < fy_new


@pytest.mark.unit
def test_temperature_window_behavior(tire_medium: TireModel) -> None:
    warm = float(tire_medium.get_grip_multiplier(95.0, 10.0))
    cold = float(tire_medium.get_grip_multiplier(60.0, 10.0))
    hot = float(tire_medium.get_grip_multiplier(135.0, 10.0))
    assert warm > cold
    assert warm > hot


@pytest.mark.unit
def test_combined_slip_reduces_available_force(tire_medium: TireModel) -> None:
    fx_pure = abs(
        float(tire_medium.calculate_longitudinal_force(0.15, 4500.0, 95.0, 5.0))
    )
    fy_pure = abs(
        float(tire_medium.calculate_lateral_force(np.deg2rad(8.0), 4500.0, 95.0, 5.0))
    )

    fx_comb, fy_comb = tire_medium.calculate_combined_forces(
        np.deg2rad(8.0),
        0.15,
        4500.0,
        95.0,
        5.0,
    )
    assert abs(float(fx_comb)) < fx_pure
    assert abs(float(fy_comb)) < fy_pure


@pytest.mark.unit
def test_self_aligning_torque_sign(tire_medium: TireModel) -> None:
    fx, fy, mz = tire_medium.calculate_forces(
        slip_angle=np.deg2rad(6.0),
        slip_ratio=0.0,
        normal_load=4500.0,
        temperature=95.0,
        wear=3.0,
        wheel_speed=210.0,
        vehicle_speed=70.0,
    )
    assert float(fy) > 0.0
    assert float(mz) < 0.0


@pytest.mark.unit
def test_temperature_rises_with_slip_energy(tire_medium: TireModel) -> None:
    initial_temp = tire_medium.state.surface_temp
    for _ in range(100):
        tire_medium.update_temperature(
            forces={"Fx": 8000.0, "Fy": 11000.0},
            speeds={
                "vx_slip": 3.5,
                "vy_slip": 2.0,
                "track_temp": 45.0,
                "camber": np.deg2rad(-3.0),
            },
            ambient_temp=30.0,
            dt=0.05,
        )
    assert tire_medium.state.surface_temp > initial_temp


@pytest.mark.unit
def test_wear_monotonic_increase(tire_medium: TireModel) -> None:
    prev = tire_medium.state.wear_percentage
    for _ in range(200):
        now = tire_medium.update_wear(np.deg2rad(6.0), 0.1, 95.0, dt=0.1)
        assert now >= prev
        prev = now


@pytest.mark.unit
def test_compound_ordering_soft_medium_hard_grip() -> None:
    soft = TireModel("soft")
    medium = TireModel("medium")
    hard = TireModel("hard")

    fy_soft = abs(
        float(soft.calculate_lateral_force(np.deg2rad(7.0), 4500.0, 100.0, 0.0))
    )
    fy_medium = abs(
        float(medium.calculate_lateral_force(np.deg2rad(7.0), 4500.0, 95.0, 0.0))
    )
    fy_hard = abs(
        float(hard.calculate_lateral_force(np.deg2rad(7.0), 4500.0, 90.0, 0.0))
    )

    assert fy_soft > fy_medium > fy_hard


@pytest.mark.unit
def test_edge_cases_no_nan_inf(tire_medium: TireModel) -> None:
    fx, fy, mz = tire_medium.calculate_forces(
        slip_angle=np.deg2rad(25.0),
        slip_ratio=2.0,
        normal_load=9000.0,
        temperature=150.0,
        wear=100.0,
        wheel_speed=500.0,
        vehicle_speed=0.0,
    )
    assert np.isfinite([fx, fy, mz]).all()


@pytest.mark.integration
def test_four_tire_integration_load_transfer() -> None:
    tires = [
        TireModel("soft"),
        TireModel("soft"),
        TireModel("medium"),
        TireModel("medium"),
    ]
    base_load = 4300.0
    transfer = TireModel.calculate_longitudinal_load_transfer(
        mass=800.0,
        acceleration_longitudinal=-8.0,
        cg_height=0.30,
        wheelbase=3.6,
    )
    loads = [
        base_load + transfer / 2.0,
        base_load + transfer / 2.0,
        base_load - transfer / 2.0,
        base_load - transfer / 2.0,
    ]

    outputs = [
        t.calculate_forces(np.deg2rad(5.0), -0.1, load, 95.0, 8.0, 220.0, 65.0)
        for t, load in zip(tires, loads)
    ]
    assert len(outputs) == 4
    assert all(np.isfinite(np.asarray(o)).all() for o in outputs)


@pytest.mark.unit
def test_vectorized_batch_operations(tire_medium: TireModel) -> None:
    n = 128
    alpha = np.deg2rad(np.linspace(-8.0, 8.0, n))
    kappa = np.linspace(-0.2, 0.2, n)
    fz = np.full(n, 4500.0)
    temp = np.full(n, 95.0)
    wear = np.linspace(0.0, 60.0, n)
    fx, fy, mz = tire_medium.calculate_forces(
        slip_angle=alpha,
        slip_ratio=kappa,
        normal_load=fz,
        temperature=temp,
        wear=wear,
        wheel_speed=np.full(n, 220.0),
        vehicle_speed=np.full(n, 70.0),
    )
    assert np.asarray(fx).shape == (n,)
    assert np.asarray(fy).shape == (n,)
    assert np.asarray(mz).shape == (n,)


@pytest.mark.slow
def test_performance_single_tire_under_target(tire_medium: TireModel) -> None:
    loops = 5000
    start = time.perf_counter()
    for _ in range(loops):
        tire_medium.calculate_forces(
            slip_angle=np.deg2rad(5.0),
            slip_ratio=0.08,
            normal_load=4500.0,
            temperature=95.0,
            wear=10.0,
            wheel_speed=210.0,
            vehicle_speed=70.0,
        )
    elapsed = time.perf_counter() - start
    per_call = elapsed / loops
    assert per_call < 0.0015


@pytest.mark.slow
def test_performance_four_tires_under_target() -> None:
    tires = [
        TireModel("soft"),
        TireModel("soft"),
        TireModel("medium"),
        TireModel("hard"),
    ]
    loops = 2000
    start = time.perf_counter()
    for _ in range(loops):
        for tire in tires:
            tire.calculate_forces(
                slip_angle=np.deg2rad(6.0),
                slip_ratio=0.1,
                normal_load=4500.0,
                temperature=95.0,
                wear=12.0,
                wheel_speed=220.0,
                vehicle_speed=72.0,
            )
    elapsed = time.perf_counter() - start
    per_step = elapsed / loops
    assert per_step < 0.005
