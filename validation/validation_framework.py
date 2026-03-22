#!/usr/bin/env python3
"""Validation framework for F1 dynamics models (Steps 2.1-2.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 1e-12:
        return 1.0
    return float(1.0 - sse / sst)


@dataclass
class ReferenceData:
    """Reference F1 data and constraints used for validation."""

    yas_marina_data: dict[str, Any]
    physics_constraints: dict[str, Any]

    @classmethod
    def build_default(cls) -> "ReferenceData":
        return cls(
            yas_marina_data={
                "lap_record": 82.109,
                "qualifying_2023": 83.5,
                "race_pace_2023": 88.0,
                "top_speed_kmh": 330.0,
                "minimum_speed_kmh": 85.0,
                "average_speed_kmh": 195.0,
                "max_lateral_g": 5.5,
                "max_longitudinal_g_accel": 1.8,
                "max_longitudinal_g_brake": 5.0,
                "sector_1": 24.8,
                "sector_2": 32.5,
                "sector_3": 24.8,
                "turn_1_speed": 130.0,
                "turn_3_speed": 195.0,
                "turn_5_hairpin": 85.0,
                "turn_11_chicane": 110.0,
                "tire_temp_range": (90.0, 110.0),
                "tire_pressure_hot": (20.5, 21.5),
                "fuel_per_lap": 1.6,
                "ers_deployment_per_lap": 4.0,
                "braking_zones": 9,
                "full_throttle_percentage": 58.0,
                "drs_zones": 2,
                "track_length_m": 5281.0,
            },
            physics_constraints={
                "max_downforce_coefficient": -7.5,
                "max_drag_coefficient": 1.2,
                "min_drag_coefficient": 0.6,
                "tire_peak_slip_angle": (8.0, 12.0),
                "tire_peak_friction": (1.8, 2.2),
                "power_to_weight": 1.0,
            },
        )


class TireModelValidator:
    """Validate tire model against known behavior and constraints."""

    def __init__(self, tire_model: Any, reference_data: ReferenceData) -> None:
        self.tire_model = tire_model
        self.reference_data = reference_data

    def validate_force_curves(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Tire Force Curves",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        test_loads = np.asarray([3000.0, 5000.0, 7000.0, 10000.0], dtype=np.float64)
        slip_angles_deg = np.linspace(-20.0, 20.0, 121)
        slip_angles_rad = np.deg2rad(slip_angles_deg)

        measured_peaks = []
        expected_peaks = []
        measured_peak_alpha = []

        for load in test_loads:
            fy = np.asarray(
                self.tire_model.calculate_lateral_force(
                    slip_angles_rad,
                    normal_load=np.full_like(slip_angles_rad, load),
                    temp=np.full_like(slip_angles_rad, 100.0),
                    wear=np.zeros_like(slip_angles_rad),
                )
            )
            abs_fy = np.abs(fy)
            peak_idx = int(np.argmax(abs_fy))
            peak_force = float(abs_fy[peak_idx])
            peak_slip = float(abs(slip_angles_deg[peak_idx]))

            expected_mu = 2.0
            expected_peak = expected_mu * float(load)
            force_error = abs(peak_force - expected_peak) / max(expected_peak, 1.0)

            measured_peaks.append(peak_force)
            expected_peaks.append(expected_peak)
            measured_peak_alpha.append(peak_slip)

            if force_error > 0.20:
                results["passed"] = False
                results["details"].append(
                    f"Load {load:.0f}N peak force error {force_error*100:.1f}% "
                    f"(measured {peak_force:.0f}N vs expected {expected_peak:.0f}N)."
                )

            expected_alpha = self.reference_data.physics_constraints[
                "tire_peak_slip_angle"
            ]
            if not (expected_alpha[0] <= peak_slip <= expected_alpha[1]):
                results["passed"] = False
                results["details"].append(
                    f"Load {load:.0f}N peak slip {peak_slip:.1f}deg outside expected {expected_alpha}."
                )

        y_true = np.asarray(expected_peaks)
        y_pred = np.asarray(measured_peaks)
        results["metrics"] = {
            "peak_force_mae": _mae(y_true, y_pred),
            "peak_force_rmse": _rmse(y_true, y_pred),
            "peak_force_r2": _r2(y_true, y_pred),
            "avg_peak_slip_deg": float(np.mean(measured_peak_alpha)),
        }
        return results

    def validate_temperature_sensitivity(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Temperature Sensitivity",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        temperatures = np.asarray([60.0, 80.0, 100.0, 120.0, 140.0], dtype=np.float64)
        grip = []

        fy_opt = abs(
            float(
                self.tire_model.calculate_lateral_force(
                    slip_angle=np.deg2rad(8.0),
                    normal_load=5000.0,
                    temp=100.0,
                    wear=0.0,
                )
            )
        )

        for temp in temperatures:
            fy = abs(
                float(
                    self.tire_model.calculate_lateral_force(
                        slip_angle=np.deg2rad(8.0),
                        normal_load=5000.0,
                        temp=float(temp),
                        wear=0.0,
                    )
                )
            )
            grip_factor = fy / max(fy_opt, 1.0)
            grip.append(grip_factor)
            if temp != 100.0 and grip_factor >= 0.985:
                results["passed"] = False
                results["details"].append(
                    f"Temp {temp:.0f}C grip factor {grip_factor:.3f} too close to optimal."
                )

        grip_arr = np.asarray(grip)
        if int(np.argmax(grip_arr)) != int(np.where(temperatures == 100.0)[0][0]):
            results["passed"] = False
            results["details"].append("Maximum grip does not occur at 100C.")

        results["metrics"] = {
            "grip_factors": {
                str(int(t)): float(g) for t, g in zip(temperatures, grip_arr)
            },
            "temp_curve_mae_to_peak": float(np.mean(np.abs(1.0 - grip_arr))),
        }
        return results

    def validate_degradation(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Tire Degradation",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        wear_levels = np.asarray([0.0, 20.0, 40.0, 60.0, 80.0, 95.0], dtype=np.float64)
        forces = []

        for wear in wear_levels:
            fy = abs(
                float(
                    self.tire_model.calculate_lateral_force(
                        slip_angle=np.deg2rad(8.0),
                        normal_load=5000.0,
                        temp=100.0,
                        wear=float(wear),
                    )
                )
            )
            forces.append(fy)

        forces_arr = np.asarray(forces)
        diffs = np.diff(forces_arr)
        if np.any(diffs > 0.0):
            results["passed"] = False
            results["details"].append(
                "Force is not monotonically decreasing with wear."
            )

        cliff_loss = (forces_arr[3] - forces_arr[-1]) / max(forces_arr[3], 1.0)
        if cliff_loss < 0.15:
            results["passed"] = False
            results["details"].append(
                f"Cliff degradation too weak ({cliff_loss*100:.1f}% loss from 60% to 95% wear)."
            )

        results["metrics"] = {
            "force_vs_wear": {
                str(int(w)): float(f) for w, f in zip(wear_levels, forces_arr)
            },
            "cliff_loss_fraction": float(cliff_loss),
        }
        return results

    def run_all_tests(self) -> dict[str, Any]:
        tests = [
            self.validate_force_curves(),
            self.validate_temperature_sensitivity(),
            self.validate_degradation(),
        ]
        return {
            "subsystem": "Tire Model",
            "all_passed": all(t["passed"] for t in tests),
            "individual_tests": tests,
        }


class AeroModelValidator:
    """Validate aerodynamics model behavior against constraints."""

    def __init__(self, aero_model: Any, reference_data: ReferenceData) -> None:
        self.aero_model = aero_model
        self.ref_data = reference_data

    def validate_downforce_magnitudes(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Downforce Magnitudes",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        speeds_kmh = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
        total_df = []
        front_pct = []

        for spd in speeds_kmh:
            spd_ms = spd / 3.6
            df_f, df_r = self.aero_model.calculate_downforce(
                speed=spd_ms,
                aero_mode="high_downforce",
                ride_height_front=0.05,
                ride_height_rear=0.055,
            )
            total = df_f + df_r
            total_df.append(total)
            fp = df_f / max(total, 1e-9)
            front_pct.append(fp)

            if spd == 300.0 and not (18000.0 <= total <= 26000.0):
                results["passed"] = False
                results["details"].append(
                    f"300 km/h downforce {total:.0f}N outside [18000, 26000]N."
                )
            if not (0.38 <= fp <= 0.47):
                results["passed"] = False
                results["details"].append(
                    f"{spd:.0f} km/h front balance {fp*100:.1f}% outside [38%, 47%]."
                )

        results["metrics"] = {
            "downforce_vs_speed": {
                str(int(s)): float(d) for s, d in zip(speeds_kmh, total_df)
            },
            "front_balance_mean": float(np.mean(front_pct)),
        }
        return results

    def validate_mode_differences(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Active Aero Modes",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        speed = 300.0 / 3.6
        df_f_h, df_r_h = self.aero_model.calculate_downforce(
            speed, "high_downforce", 0.05, 0.055
        )
        drag_h = self.aero_model.calculate_total_drag(speed, "high_downforce")

        df_f_l, df_r_l = self.aero_model.calculate_downforce(
            speed, "low_drag", 0.05, 0.055
        )
        drag_l = self.aero_model.calculate_total_drag(speed, "low_drag")

        df_ratio = (df_f_l + df_r_l) / max(df_f_h + df_r_h, 1.0)
        drag_ratio = drag_l / max(drag_h, 1.0)

        if not (0.45 <= df_ratio <= 0.60):
            results["passed"] = False
            results["details"].append(
                f"Downforce ratio low/high {df_ratio:.2f} outside [0.45, 0.60]."
            )

        if not (0.55 <= drag_ratio <= 0.70):
            results["passed"] = False
            results["details"].append(
                f"Drag ratio low/high {drag_ratio:.2f} outside [0.55, 0.70]."
            )

        results["metrics"] = {
            "downforce_ratio": float(df_ratio),
            "drag_ratio": float(drag_ratio),
        }
        return results

    def validate_ground_effect(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "test_name": "Ground Effect Sensitivity",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        ride_heights = np.asarray([0.025, 0.040, 0.055, 0.070], dtype=np.float64)
        downforces = []
        for rh in ride_heights:
            df_f, df_r = self.aero_model.calculate_downforce(
                speed=250.0 / 3.6,
                aero_mode="high_downforce",
                ride_height_front=float(rh),
                ride_height_rear=float(rh),
            )
            downforces.append(df_f + df_r)

        down_arr = np.asarray(downforces)
        rh_opt = float(ride_heights[int(np.argmax(down_arr))])

        if not (0.035 <= rh_opt <= 0.060):
            results["passed"] = False
            results["details"].append(
                f"Optimal ride height {rh_opt:.3f}m outside [0.035, 0.060]m."
            )

        if down_arr[0] > down_arr[1]:
            results["passed"] = False
            results["details"].append(
                "Lowest ride height creates more downforce than near-optimal, stall behavior missing."
            )

        results["metrics"] = {
            "downforce_vs_ride_height": {
                f"{rh:.3f}": float(df) for rh, df in zip(ride_heights, down_arr)
            },
            "optimal_ride_height": rh_opt,
        }
        return results

    def run_all_tests(self) -> dict[str, Any]:
        tests = [
            self.validate_downforce_magnitudes(),
            self.validate_mode_differences(),
            self.validate_ground_effect(),
        ]
        return {
            "subsystem": "Aerodynamics Model",
            "all_passed": all(t["passed"] for t in tests),
            "individual_tests": tests,
        }


class PowertrainValidator:
    """Validate powertrain outputs and constraints."""

    def __init__(self, powertrain_model: Any) -> None:
        self.powertrain = powertrain_model

    def validate_power_output(self) -> dict[str, Any]:
        results = {
            "test_name": "Power Output",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.powertrain.reset(110.0)
        self.powertrain.battery.state.state_of_charge = 0.8
        self.powertrain.battery.state.energy_j = 0.8 * self.powertrain.battery.capacity

        for _ in range(20):
            self.powertrain.update(0.02)
            _, _, _, st = self.powertrain.calculate_wheel_power(
                throttle=1.0,
                brake=0.0,
                rpm=11000.0,
                gear=6,
                vehicle_speed=80.0,
                deployment_strategy="max_power",
                harvest_strategy="balanced",
                gap_to_leader=None,
                dt=0.02,
            )

        total_power = float(st["total_power_w"])
        if not (700000.0 <= total_power <= 800000.0):
            results["passed"] = False
            results["details"].append(
                f"Total power {total_power/1000.0:.1f}kW outside [700, 800]kW."
            )

        results["metrics"] = {
            "total_power_kw": total_power / 1000.0,
            "ice_power_kw": float(st["ice_power_w"]) / 1000.0,
            "mguk_power_kw": float(st["mguk_power_w"]) / 1000.0,
        }
        return results

    def validate_power_split(self) -> dict[str, Any]:
        results = {
            "test_name": "50:50 Power Split",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        ice_power, _, _ = self.powertrain.ice.calculate_ice_power(
            rpm=11000.0, throttle=1.0, dt=0.01
        )
        mguk_power, _, _ = self.powertrain.mguk.calculate_motor_power(
            deployment_rate=1.0, rpm=11000.0
        )
        total = ice_power + mguk_power
        ice_frac = ice_power / max(total, 1.0)
        mguk_frac = mguk_power / max(total, 1.0)

        if not (0.48 <= ice_frac <= 0.52):
            results["passed"] = False
            results["details"].append(
                f"ICE fraction {ice_frac:.3f} outside [0.48, 0.52]."
            )

        if not (0.48 <= mguk_frac <= 0.52):
            results["passed"] = False
            results["details"].append(
                f"MGU-K fraction {mguk_frac:.3f} outside [0.48, 0.52]."
            )

        results["metrics"] = {
            "ice_fraction": float(ice_frac),
            "electric_fraction": float(mguk_frac),
        }
        return results

    def validate_energy_recovery(self) -> dict[str, Any]:
        results = {
            "test_name": "Energy Recovery",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        regen_torque, battery_charge = self.powertrain.calculate_energy_recovery(
            braking=1.0,
            rpm=12000.0,
            vehicle_speed=80.0,
            harvest_strategy="aggressive",
            throttle=0.0,
        )
        _ = regen_torque

        expected = 350000.0
        if battery_charge > expected * 1.1:
            results["passed"] = False
            results["details"].append(
                f"Recovery {battery_charge/1000.0:.1f}kW above 350kW limit envelope."
            )
        if battery_charge < expected * 0.7:
            results["passed"] = False
            results["details"].append(
                f"Recovery {battery_charge/1000.0:.1f}kW lower than expected aggressive range."
            )

        results["metrics"] = {
            "battery_charge_kw": float(battery_charge) / 1000.0,
        }
        return results

    def run_all_tests(self) -> dict[str, Any]:
        tests = [
            self.validate_power_output(),
            self.validate_power_split(),
            self.validate_energy_recovery(),
        ]
        return {
            "subsystem": "Powertrain Model",
            "all_passed": all(t["passed"] for t in tests),
            "individual_tests": tests,
        }


class VehicleDynamicsValidator:
    """Validate complete vehicle dynamics behavior."""

    def __init__(
        self, vehicle_model: Any, reference_data: ReferenceData, fast_mode: bool = False
    ) -> None:
        self.vehicle = vehicle_model
        self.ref_data = reference_data
        self.fast_mode = fast_mode

    def validate_acceleration(self) -> dict[str, Any]:
        results = {
            "test_name": "Acceleration Performance",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.vehicle.reset(fuel_load=110.0)
        dt = 0.01
        max_time = 8.0 if self.fast_mode else 20.0
        t = 0.0

        t100 = None
        t200 = None
        t300 = None

        while t < max_time:
            spd = self.vehicle.get_speed_kmh()
            if t100 is None and spd >= 100.0:
                t100 = t
            if t200 is None and spd >= 200.0:
                t200 = t
            if t300 is None and spd >= 300.0:
                t300 = t
                break

            self.vehicle.update(
                dt=dt,
                steering=0.0,
                throttle=1.0,
                brake=0.0,
                aero_mode="low_drag",
            )
            t += dt

        if t100 is None:
            results["passed"] = False
            results["details"].append("Failed to reach 100 km/h within test horizon.")
        elif not (1.8 <= t100 <= 3.2):
            results["passed"] = False
            results["details"].append(f"0-100 km/h = {t100:.2f}s outside [1.8, 3.2]s.")

        if t200 is not None and not (4.0 <= t200 <= 8.0):
            results["passed"] = False
            results["details"].append(f"0-200 km/h = {t200:.2f}s outside [4.0, 8.0]s.")

        results["metrics"] = {
            "time_to_100": t100,
            "time_to_200": t200,
            "time_to_300": t300,
        }
        return results

    def validate_braking(self) -> dict[str, Any]:
        results = {
            "test_name": "Braking Performance",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.vehicle.reset(fuel_load=50.0)
        self.vehicle.state.vx = 300.0 / 3.6

        dt = 0.005
        max_decel = 0.0
        steps = 120 if self.fast_mode else 300
        for _ in range(steps):
            self.vehicle.update(dt, 0.0, 0.0, 1.0, "high_downforce")
            decel_g = max(0.0, -self.vehicle.state.ax / 9.81)
            max_decel = max(max_decel, decel_g)

        if not (2.0 <= max_decel <= 6.0):
            results["passed"] = False
            results["details"].append(
                f"Max braking decel {max_decel:.2f}G outside [2.0, 6.0]G."
            )

        results["metrics"] = {"max_braking_g": float(max_decel)}
        return results

    def validate_cornering(self) -> dict[str, Any]:
        results = {
            "test_name": "Cornering Performance",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.vehicle.reset(fuel_load=70.0)
        self.vehicle.state.vx = 200.0 / 3.6

        dt = 0.01
        steps = 150 if self.fast_mode else 400
        for _ in range(steps):
            self.vehicle.update(dt, np.deg2rad(5.0), 0.7, 0.0, "high_downforce")

        lateral_g = abs(self.vehicle.state.ay) / 9.81
        if not (0.8 <= lateral_g <= 6.5):
            results["passed"] = False
            results["details"].append(
                f"Steady corner lateral G {lateral_g:.2f} outside [0.8, 6.5]."
            )

        results["metrics"] = {"lateral_g": float(lateral_g)}
        return results

    def validate_top_speed(self) -> dict[str, Any]:
        results = {
            "test_name": "Top Speed",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.vehicle.reset(fuel_load=40.0)
        dt = 0.02
        max_time = 25.0 if self.fast_mode else 90.0
        top_speed = 0.0
        prev = 0.0

        for _ in range(int(max_time / dt)):
            self.vehicle.update(dt, 0.0, 1.0, 0.0, "low_drag")
            speed = self.vehicle.get_speed_kmh()
            top_speed = max(top_speed, speed)
            if abs(speed - prev) < 0.03 and speed > 40.0:
                break
            prev = speed

        if not (250.0 <= top_speed <= 365.0):
            results["passed"] = False
            results["details"].append(
                f"Top speed {top_speed:.1f}km/h outside [250, 365]km/h."
            )

        results["metrics"] = {"top_speed_kmh": float(top_speed)}
        return results

    def run_all_tests(self) -> dict[str, Any]:
        tests = [
            self.validate_acceleration(),
            self.validate_braking(),
            self.validate_cornering(),
            self.validate_top_speed(),
        ]
        return {
            "subsystem": "Vehicle Dynamics",
            "all_passed": all(t["passed"] for t in tests),
            "individual_tests": tests,
        }


class LapSimulationValidator:
    """Validate coarse lap simulation against reference pace envelope."""

    def __init__(
        self, vehicle_model: Any, reference_data: ReferenceData, fast_mode: bool = False
    ) -> None:
        self.vehicle = vehicle_model
        self.reference_data = reference_data
        self.fast_mode = fast_mode

    def simulate_optimal_lap(self) -> dict[str, Any]:
        results = {
            "test_name": "Lap Time Simulation",
            "passed": True,
            "details": [],
            "metrics": {},
        }

        self.vehicle.reset(fuel_load=90.0)
        dt = 0.01
        distance = 0.0
        lap_time = 0.0
        track_length = float(self.reference_data.yas_marina_data["track_length_m"])
        if self.fast_mode:
            track_length *= 0.2

        sector_marks = [1760.0, 4410.0, track_length]
        sector_times = [0.0, 0.0, 0.0]
        sector_idx = 0

        lap_timeout = 45.0 if self.fast_mode else 220.0
        while lap_time < lap_timeout and distance < track_length:
            # Simple heuristics to stress model behavior for validation.
            pos_ratio = distance / max(track_length, 1.0)
            steering = np.deg2rad(3.5 * np.sin(2.0 * np.pi * pos_ratio * 4.0))
            brake = max(0.0, np.sin(2.0 * np.pi * pos_ratio * 9.0 - 1.1))
            throttle = float(np.clip(0.88 - 0.75 * brake, 0.0, 1.0))
            aero_mode = (
                "high_downforce" if abs(steering) > np.deg2rad(1.5) else "low_drag"
            )

            self.vehicle.update(dt, steering, throttle, brake, aero_mode)
            distance += max(self.vehicle.state.vx, 0.0) * dt
            lap_time += dt

            while (
                sector_idx < len(sector_marks) and distance >= sector_marks[sector_idx]
            ):
                prev = sum(sector_times[:sector_idx])
                sector_times[sector_idx] = lap_time - prev
                sector_idx += 1

        ref_lap = float(self.reference_data.yas_marina_data["lap_record"])
        lower = ref_lap
        upper = ref_lap * 1.35
        if not (lower <= lap_time <= upper):
            results["passed"] = False
            results["details"].append(
                f"Lap time {lap_time:.2f}s outside [{lower:.2f}, {upper:.2f}]s validation envelope."
            )

        results["metrics"] = {
            "lap_time_s": float(lap_time),
            "sector_times_s": [float(v) for v in sector_times],
            "distance_traveled_m": float(distance),
        }
        results["details"].append(f"Simulated lap time: {lap_time:.2f}s")
        return results


class ValidationFramework:
    """Validation and calibration orchestration for vehicle dynamics stack."""

    def __init__(
        self,
        vehicle_model: Any,
        reference_data: ReferenceData | None = None,
        fast_mode: bool = False,
    ) -> None:
        self.vehicle = vehicle_model
        self.reference_data = reference_data or ReferenceData.build_default()
        self.fast_mode = fast_mode

        self.tire_validator = TireModelValidator(
            self.vehicle.tire_models[0], self.reference_data
        )
        self.aero_validator = AeroModelValidator(
            self.vehicle.aero_model, self.reference_data
        )
        self.powertrain_validator = PowertrainValidator(self.vehicle.powertrain)
        self.vehicle_validator = VehicleDynamicsValidator(
            self.vehicle, self.reference_data, fast_mode=fast_mode
        )
        self.lap_validator = LapSimulationValidator(
            self.vehicle, self.reference_data, fast_mode=fast_mode
        )

    def _confidence_score(self, subsystem_result: dict[str, Any]) -> float:
        tests = subsystem_result["individual_tests"]
        if not tests:
            return 0.0
        pass_ratio = np.mean([1.0 if t["passed"] else 0.0 for t in tests])

        penalty = 0.0
        for test in tests:
            details = test.get("details", [])
            penalty += min(len(details) * 0.03, 0.2)
        return float(np.clip(pass_ratio - penalty, 0.0, 1.0))

    def _sensitivity_analysis(self) -> dict[str, Any]:
        """Finite-difference parameter sensitivity around nominal conditions."""
        sensitivities: dict[str, float] = {}

        tire = self.vehicle.tire_models[0]
        base_fy = abs(
            float(
                tire.calculate_lateral_force(
                    slip_angle=np.deg2rad(8.0),
                    normal_load=5000.0,
                    temp=100.0,
                    wear=5.0,
                )
            )
        )
        lat_cfg = tire.params["lateral_force"]
        for key in ["B", "C", "D_nominal", "E"]:
            original = float(lat_cfg[key])
            step = 0.02 * (abs(original) if abs(original) > 1e-6 else 1.0)
            lat_cfg[key] = original + step
            pert_fy = abs(
                float(
                    tire.calculate_lateral_force(
                        slip_angle=np.deg2rad(8.0),
                        normal_load=5000.0,
                        temp=100.0,
                        wear=5.0,
                    )
                )
            )
            sensitivities[f"tire_{key}"] = float((pert_fy - base_fy) / max(step, 1e-9))
            lat_cfg[key] = original

        base_drag = self.vehicle.aero_model.calculate_total_drag(
            300.0 / 3.6, "high_downforce"
        )
        cd = float(self.vehicle.aero_model.modes["high_downforce"]["CD_base"])
        step_cd = 0.01
        self.vehicle.aero_model.modes["high_downforce"]["CD_base"] = cd + step_cd
        pert_drag = self.vehicle.aero_model.calculate_total_drag(
            300.0 / 3.6, "high_downforce"
        )
        sensitivities["aero_CD_base_high"] = float((pert_drag - base_drag) / step_cd)
        self.vehicle.aero_model.modes["high_downforce"]["CD_base"] = cd

        return {
            "local_gradients": sensitivities,
            "most_sensitive": sorted(
                sensitivities, key=lambda k: abs(sensitivities[k]), reverse=True
            )[:5],
        }

    def run_complete_validation(self, include_lap: bool = True) -> dict[str, Any]:
        """Run all subsystem tests and attach confidence/statistics."""
        results: dict[str, Any] = {}

        results["Tire Model"] = self.tire_validator.run_all_tests()
        results["Aerodynamics Model"] = self.aero_validator.run_all_tests()
        results["Powertrain Model"] = self.powertrain_validator.run_all_tests()
        results["Vehicle Dynamics"] = self.vehicle_validator.run_all_tests()
        if include_lap:
            results["Lap Simulation"] = {
                "subsystem": "Lap Simulation",
                "all_passed": True,
                "individual_tests": [self.lap_validator.simulate_optimal_lap()],
            }
            results["Lap Simulation"]["all_passed"] = all(
                t["passed"] for t in results["Lap Simulation"]["individual_tests"]
            )

        confidence = {
            subsystem: self._confidence_score(result)
            for subsystem, result in results.items()
        }

        summary = {
            "all_passed": all(r["all_passed"] for r in results.values()),
            "subsystems_passed": int(
                sum(1 for r in results.values() if r["all_passed"])
            ),
            "subsystems_total": int(len(results)),
            "confidence": confidence,
            "sensitivity": self._sensitivity_analysis(),
        }

        return {
            "summary": summary,
            "results": results,
            "reference": {
                "yas_marina": self.reference_data.yas_marina_data,
                "physics_constraints": self.reference_data.physics_constraints,
            },
        }
