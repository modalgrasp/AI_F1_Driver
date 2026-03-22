#!/usr/bin/env python3
"""Tests for 2026 F1 active aerodynamics model."""

from __future__ import annotations

import time

import numpy as np
import pytest

from vehicle_dynamics.aerodynamics import AerodynamicsModel


@pytest.fixture
def aero_model() -> AerodynamicsModel:
    return AerodynamicsModel()


@pytest.mark.unit
def test_downforce_scales_with_speed_squared(aero_model: AerodynamicsModel) -> None:
    df1 = sum(
        aero_model.calculate_forces(
            speed=150.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )[:2]
    )
    df2 = sum(
        aero_model.calculate_forces(
            speed=300.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )[:2]
    )
    ratio = df2 / max(df1, 1.0)
    assert 3.5 < ratio < 4.5


@pytest.mark.unit
def test_mode_force_magnitudes_reference(aero_model: AerodynamicsModel) -> None:
    high = aero_model.calculate_forces(
        speed=300.0 / 3.6,
        aero_mode="high_downforce",
        ride_height_front=0.05,
        ride_height_rear=0.055,
        pitch=0.0,
        roll=0.0,
        yaw=0.0,
    )
    aero_model.set_aero_mode("low_drag")
    for _ in range(20):
        aero_model.update(0.01)
    low = aero_model.calculate_forces(
        speed=300.0 / 3.6,
        aero_mode="low_drag",
        ride_height_front=0.05,
        ride_height_rear=0.055,
        pitch=0.0,
        roll=0.0,
        yaw=0.0,
    )

    high_downforce = high[0] + high[1]
    low_downforce = low[0] + low[1]
    high_drag = high[2]
    low_drag = low[2]

    assert 15000.0 < high_downforce < 32000.0
    assert 9000.0 < low_downforce < 22000.0
    assert high_downforce > low_downforce
    assert high_drag > low_drag


@pytest.mark.unit
def test_ground_effect_height_sensitivity(aero_model: AerodynamicsModel) -> None:
    low_height = sum(
        aero_model.calculate_forces(
            speed=250.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.04,
            ride_height_rear=0.045,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )[:2]
    )
    high_height = sum(
        aero_model.calculate_forces(
            speed=250.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.09,
            ride_height_rear=0.095,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )[:2]
    )
    assert low_height > high_height


@pytest.mark.unit
def test_dirty_air_reduces_downforce(aero_model: AerodynamicsModel) -> None:
    clean = sum(
        aero_model.calculate_forces(
            speed=270.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            distance_to_car_ahead=None,
        )[:2]
    )
    dirty = sum(
        aero_model.calculate_forces(
            speed=270.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            distance_to_car_ahead=5.0,
        )[:2]
    )
    assert dirty < clean


@pytest.mark.unit
def test_mode_switching_transition_is_smooth(aero_model: AerodynamicsModel) -> None:
    aero_model.reset()
    aero_model.set_aero_mode("low_drag")

    history = []
    for _ in range(20):
        aero_model.update(0.01)
        forces = aero_model.calculate_forces(
            speed=260.0 / 3.6,
            aero_mode="low_drag",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
        )
        history.append(forces[2])

    diffs = np.diff(np.asarray(history))
    assert np.max(np.abs(diffs)) < 2500.0


@pytest.mark.unit
def test_pitch_roll_yaw_moment_signs(aero_model: AerodynamicsModel) -> None:
    _, _, _, pitch_m, roll_m, yaw_m = aero_model.calculate_forces(
        speed=220.0 / 3.6,
        aero_mode="high_downforce",
        ride_height_front=0.05,
        ride_height_rear=0.055,
        pitch=np.deg2rad(0.5),
        roll=np.deg2rad(1.0),
        yaw=np.deg2rad(2.0),
    )
    assert np.isfinite([pitch_m, roll_m, yaw_m]).all()
    assert yaw_m < 0.0


@pytest.mark.unit
def test_air_density_reasonable_range() -> None:
    rho = AerodynamicsModel.calculate_air_density(
        temperature=30.0, pressure=101325.0, humidity=0.4
    )
    assert 1.0 < rho < 1.3


@pytest.mark.integration
def test_integration_with_four_tire_load_case(aero_model: AerodynamicsModel) -> None:
    df_f, df_r, drag, *_ = aero_model.calculate_forces(
        speed=290.0 / 3.6,
        aero_mode="high_downforce",
        ride_height_front=0.05,
        ride_height_rear=0.055,
        pitch=np.deg2rad(-0.3),
        roll=np.deg2rad(0.5),
        yaw=np.deg2rad(0.4),
    )
    assert df_f > 0.0
    assert df_r > 0.0
    assert drag > 0.0


@pytest.mark.slow
def test_performance_under_target(aero_model: AerodynamicsModel) -> None:
    loops = 8000
    start = time.perf_counter()
    for _ in range(loops):
        aero_model.calculate_forces(
            speed=260.0 / 3.6,
            aero_mode="high_downforce",
            ride_height_front=0.05,
            ride_height_rear=0.055,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            distance_to_car_ahead=7.0,
        )
    elapsed = time.perf_counter() - start
    per_call = elapsed / loops
    assert per_call < 0.001
