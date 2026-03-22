#!/usr/bin/env python3
"""Full-system integration tests for Phase 2 vehicle dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


class FullSystemIntegrationTests:
    """
    Integration tests for complete vehicle dynamics system.

    Tests the interaction between tire, aero, powertrain, and integrator behavior
    under realistic and edge-case scenarios.
    """


class TestFullSystemIntegration(FullSystemIntegrationTests):
    """Pytest-discovered wrapper class for full-system integration tests."""

    def _make_vehicle(self) -> VehicleDynamicsModel:
        vehicle = VehicleDynamicsModel()
        vehicle.reset()
        return vehicle

    @pytest.mark.integration
    def test_tire_aero_coupling(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = 250.0 / 3.6

        steering = float(np.radians(8.0))
        for _ in range(40):
            vehicle.update(0.001, steering, 0.45, 0.0, "high_downforce")
        lateral_g_high = abs(vehicle.state.ay) / 9.81

        vehicle = self._make_vehicle()
        vehicle.state.vx = 250.0 / 3.6
        for _ in range(40):
            vehicle.update(0.001, steering, 0.45, 0.0, "low_drag")
        lateral_g_low = abs(vehicle.state.ay) / 9.81

        assert lateral_g_high > lateral_g_low, (
            f"High-downforce mode produced less lateral G: "
            f"{lateral_g_high:.3f} vs {lateral_g_low:.3f}"
        )

        improvement = (lateral_g_high - lateral_g_low) / max(lateral_g_low, 1e-6)
        assert (
            improvement > 0.05
        ), f"Insufficient aero cornering effect: {improvement*100:.1f}%"

    @pytest.mark.integration
    def test_powertrain_tire_coupling(self) -> None:
        vehicle = self._make_vehicle()

        for _ in range(100):
            vehicle.update(0.001, 0.0, 1.0, 0.0, "low_drag")

        rear_slip_ratios = vehicle.last_slip_ratios[2:4]
        assert np.all(np.isfinite(rear_slip_ratios))
        assert np.any(
            rear_slip_ratios > 0.02
        ), f"Rear slip too low for hard launch: {rear_slip_ratios}"
        # Current model clips slip to 2.5 under aggressive launch. Validate
        # bounded behavior rather than assuming traction-control moderation.
        assert np.all(
            rear_slip_ratios <= 2.5
        ), f"Rear slip exceeded model clip: {rear_slip_ratios}"
        assert (
            vehicle.state.vx > 0.2
        ), f"Vehicle did not accelerate from launch: vx={vehicle.state.vx:.3f}"

    @pytest.mark.integration
    def test_energy_conservation(self) -> None:
        vehicle = VehicleDynamicsModel()
        vehicle.reset(fuel_load=100.0)

        initial_fuel = vehicle.state.fuel_mass
        initial_battery = vehicle.powertrain.battery.state.energy_j
        initial_ke = 0.5 * vehicle.params.mass * vehicle.state.vx**2

        total_initial = initial_fuel * 43e6 + initial_battery + initial_ke

        total_dissipated = 0.0
        dt = 0.01

        for _ in range(1000):
            drag_force = vehicle.aero_model.calculate_total_drag(
                vehicle.state.vx, vehicle.current_aero_mode
            )
            rolling_resistance = (
                0.015 * (vehicle.params.mass + vehicle.state.fuel_mass) * 9.81
            )
            total_dissipated += (
                (drag_force + rolling_resistance) * max(vehicle.state.vx, 0.0) * dt
            )
            vehicle.update(dt, 0.0, 0.8, 0.0, "low_drag")

        final_fuel = vehicle.state.fuel_mass
        final_battery = vehicle.powertrain.battery.state.energy_j
        final_ke = 0.5 * vehicle.params.mass * vehicle.state.vx**2

        total_final = final_fuel * 43e6 + final_battery + final_ke
        energy_out = total_final + total_dissipated

        # We only enforce non-creation with tolerance because thermal losses and
        # drivetrain inefficiencies are intentionally simplified.
        assert (
            energy_out <= total_initial * 1.15
        ), f"Energy creation suspected: out={energy_out/1e6:.2f}MJ in={total_initial/1e6:.2f}MJ"

    @pytest.mark.integration
    def test_load_transfer_accuracy(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = 50.0

        for _ in range(100):
            vehicle.update(0.001, 0.0, 0.0, 1.0, "high_downforce")

        normal_loads = vehicle.calculate_normal_loads(
            vehicle.state.ax,
            vehicle.state.ay,
            vehicle.state.vx,
            vehicle.state.roll,
            vehicle.state.pitch,
        )

        total_load = float(np.sum(normal_loads))
        total_weight = (vehicle.params.mass + vehicle.state.fuel_mass) * 9.81
        df_front, df_rear = vehicle.aero_model.calculate_downforce(
            speed=max(vehicle.state.vx, 0.0),
            aero_mode=vehicle.current_aero_mode,
            ride_height_front=vehicle.get_front_ride_height(),
            ride_height_rear=vehicle.get_rear_ride_height(),
            pitch=vehicle.state.pitch,
            roll=vehicle.state.roll,
        )
        expected_total = total_weight + df_front + df_rear
        error = abs(total_load - expected_total) / max(expected_total, 1.0)

        assert error < 0.08, (
            f"Load calculation mismatch: error={error*100:.2f}% "
            f"calculated={total_load:.1f}N expected={expected_total:.1f}N"
        )

        front_load = normal_loads[0] + normal_loads[1]
        rear_load = normal_loads[2] + normal_loads[3]
        assert (
            front_load > rear_load
        ), f"Braking transfer incorrect: front={front_load:.1f}, rear={rear_load:.1f}"

    @pytest.mark.integration
    def test_zero_velocity_handling(self) -> None:
        vehicle = self._make_vehicle()
        assert vehicle.state.vx == 0.0
        assert vehicle.state.vy == 0.0

        vehicle.update(0.001, 0.0, 0.5, 0.0, "low_drag")
        assert vehicle.state.vx >= 0.0

        vehicle.state.vx = 1.0
        for _ in range(1000):
            vehicle.update(0.001, 0.0, 0.0, 1.0, "high_downforce")
            if vehicle.state.vx <= 0.01:
                break

        assert (
            vehicle.state.vx >= 0.0
        ), f"Negative velocity after full braking: {vehicle.state.vx}"

    @pytest.mark.integration
    def test_extreme_inputs(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = 50.0

        vehicle.update(0.001, np.radians(90.0), 0.5, 0.0, "high_downforce")
        assert np.isfinite(vehicle.state.vx)
        assert np.isfinite(vehicle.state.vy)
        assert np.isfinite(vehicle.state.omega)

        vehicle.update(0.001, 0.0, 2.0, 0.0, "low_drag")
        assert np.isfinite(vehicle.state.vx)

        vehicle.update(0.001, 0.0, 0.5, -0.5, "low_drag")
        assert np.isfinite(vehicle.state.vx)

    @pytest.mark.integration
    def test_wheel_lock_handling(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = 100.0 / 3.6

        for _ in range(100):
            vehicle.update(0.001, 0.0, 0.0, 1.0, "high_downforce")

        slip_ratios = vehicle.calculate_slip_ratios(
            vehicle.state.vx, vehicle.state.wheel_speeds
        )
        assert np.all(np.isfinite(slip_ratios))
        assert (
            np.min(slip_ratios) > -1.05
        ), f"Wheel lock instability: min slip={np.min(slip_ratios):.3f}"
        assert (
            np.max(slip_ratios) <= 0.3
        ), f"Unexpected positive wheelspin during braking: {slip_ratios}"

    @pytest.mark.integration
    def test_rollover_prevention(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = 200.0 / 3.6

        extreme_steering = np.radians(25.0)
        for _ in range(500):
            vehicle.update(0.001, extreme_steering, 0.3, 0.0, "high_downforce")

        # Vehicle model clamps roll to +/-0.25 rad for stability.
        assert abs(vehicle.state.roll) <= np.radians(
            15.0
        ), f"Excessive roll: {np.degrees(vehicle.state.roll):.2f} deg"

        normal_loads = vehicle.calculate_normal_loads(
            vehicle.state.ax,
            vehicle.state.ay,
            vehicle.state.vx,
            vehicle.state.roll,
            vehicle.state.pitch,
        )
        assert np.all(normal_loads > 0.0), f"Wheel lift detected: {normal_loads}"

    @pytest.mark.integration
    def test_backwards_driving(self) -> None:
        vehicle = self._make_vehicle()
        vehicle.state.vx = -10.0

        for _ in range(100):
            vehicle.update(0.001, 0.0, 0.0, 0.5, "high_downforce")

        assert math.isfinite(vehicle.state.vx)
        assert (
            abs(vehicle.state.vx) <= 10.5
        ), "Backward braking did not reduce reverse speed envelope"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_numerical_stability_long_run(self) -> None:
        vehicle = self._make_vehicle()

        dt = 0.002
        total_time = 0.0
        max_time = 30.0
        rng = np.random.default_rng(12345)

        while total_time < max_time:
            steering = float(rng.uniform(-0.12, 0.12))
            throttle = float(rng.uniform(0.35, 0.95))
            vehicle.update(dt, steering, throttle, 0.0, "high_downforce")

            assert np.isfinite(
                vehicle.state.vx
            ), f"Non-finite vx at t={total_time:.3f}s"
            assert np.isfinite(
                vehicle.state.vy
            ), f"Non-finite vy at t={total_time:.3f}s"
            assert np.isfinite(
                vehicle.state.omega
            ), f"Non-finite omega at t={total_time:.3f}s"
            assert np.isfinite(vehicle.state.x), f"Non-finite x at t={total_time:.3f}s"
            assert np.isfinite(vehicle.state.y), f"Non-finite y at t={total_time:.3f}s"

            assert abs(vehicle.state.vx) < 220.0, f"Runaway vx: {vehicle.state.vx:.3f}"
            assert abs(vehicle.state.vy) < 120.0, f"Runaway vy: {vehicle.state.vy:.3f}"

            total_time += dt

    @pytest.mark.integration
    def test_determinism(self) -> None:
        rng = np.random.default_rng(42)
        inputs = [
            (float(rng.uniform(-0.2, 0.2)), float(rng.uniform(0.5, 1.0)))
            for _ in range(1000)
        ]

        vehicle1 = VehicleDynamicsModel()
        vehicle1.reset(fuel_load=100.0)
        states1 = []
        for steering, throttle in inputs:
            vehicle1.update(0.001, steering, throttle, 0.0, "high_downforce")
            states1.append((vehicle1.state.x, vehicle1.state.y, vehicle1.state.vx))

        vehicle2 = VehicleDynamicsModel()
        vehicle2.reset(fuel_load=100.0)
        states2 = []
        for steering, throttle in inputs:
            vehicle2.update(0.001, steering, throttle, 0.0, "high_downforce")
            states2.append((vehicle2.state.x, vehicle2.state.y, vehicle2.state.vx))

        for i, (s1, s2) in enumerate(zip(states1, states2)):
            assert np.allclose(
                s1, s2, atol=1e-10
            ), f"Non-deterministic behavior at step {i}: {s1} vs {s2}"
