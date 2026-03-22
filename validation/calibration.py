#!/usr/bin/env python3
"""Parameter calibration utilities for vehicle dynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class CalibrationResult:
    parameters: dict[str, float]
    objective: float
    method: str
    success: bool


class ParameterCalibrator:
    """Automatic parameter tuning against reference performance targets."""

    def __init__(self, vehicle_model: Any, reference_data: Any) -> None:
        self.vehicle = vehicle_model
        self.reference = reference_data

    def _safe_optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float, bool, str]:
        try:
            from scipy.optimize import minimize

            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80},
            )
            return (
                np.asarray(res.x, dtype=np.float64),
                float(res.fun),
                bool(res.success),
                "L-BFGS-B",
            )
        except Exception:
            # Fallback random search when scipy is unavailable.
            best = np.asarray(x0, dtype=np.float64)
            best_val = float(objective(best))
            rng = np.random.default_rng(42)
            for _ in range(80):
                cand = np.asarray(
                    [rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64
                )
                val = float(objective(cand))
                if val < best_val:
                    best, best_val = cand, val
            return best, best_val, True, "random-search"

    def test_acceleration_to_100(self) -> float:
        self.vehicle.reset(fuel_load=110.0)
        dt = 0.01
        t = 0.0
        while t < 25.0:
            self.vehicle.update(
                dt, steering=0.0, throttle=1.0, brake=0.0, aero_mode="low_drag"
            )
            if self.vehicle.get_speed_kmh() >= 100.0:
                return t
            t += dt
        return 25.0

    def test_steady_corner(self) -> float:
        self.vehicle.reset(fuel_load=80.0)
        self.vehicle.state.vx = 180.0 / 3.6
        for _ in range(300):
            self.vehicle.update(
                0.01,
                steering=np.deg2rad(5.0),
                throttle=0.65,
                brake=0.0,
                aero_mode="high_downforce",
            )
        return abs(self.vehicle.state.ay) / 9.81

    def calibrate_tire_parameters(self) -> CalibrationResult:
        tire = self.vehicle.tire_models[0]
        lat_cfg = tire.params["lateral_force"]

        base = np.asarray(
            [
                float(lat_cfg["B"]),
                float(lat_cfg["C"]),
                float(lat_cfg["D_nominal"]),
                float(lat_cfg["E"]),
            ],
            dtype=np.float64,
        )

        bounds = [
            (8.0, 15.0),
            (1.2, 1.5),
            (1.6, 2.4),
            (-1.0, 0.0),
        ]

        def objective(x: np.ndarray) -> float:
            lat_cfg["B"] = float(x[0])
            lat_cfg["C"] = float(x[1])
            lat_cfg["D_nominal"] = float(x[2])
            lat_cfg["E"] = float(x[3])

            t100 = self.test_acceleration_to_100()
            lat_g = self.test_steady_corner()
            return float((t100 - 2.0) ** 2 + (lat_g - 5.0) ** 2)

        x_best, f_best, ok, method = self._safe_optimize(objective, base, bounds)
        lat_cfg["B"] = float(x_best[0])
        lat_cfg["C"] = float(x_best[1])
        lat_cfg["D_nominal"] = float(x_best[2])
        lat_cfg["E"] = float(x_best[3])

        return CalibrationResult(
            parameters={
                "B_lateral": float(x_best[0]),
                "C_lateral": float(x_best[1]),
                "D_coefficient": float(x_best[2]),
                "E_lateral": float(x_best[3]),
            },
            objective=float(f_best),
            method=method,
            success=ok,
        )

    def calibrate_aero_parameters(self) -> CalibrationResult:
        modes = self.vehicle.aero_model.modes
        x0 = np.asarray(
            [
                float(modes["high_downforce"]["CL_front"]),
                float(modes["high_downforce"]["CL_rear"]),
                float(modes["high_downforce"]["CD_base"]),
            ],
            dtype=np.float64,
        )
        bounds = [(-4.5, -2.0), (-4.8, -2.0), (0.9, 1.3)]

        def objective(x: np.ndarray) -> float:
            modes["high_downforce"]["CL_front"] = float(x[0])
            modes["high_downforce"]["CL_rear"] = float(x[1])
            modes["high_downforce"]["CD_base"] = float(x[2])

            df_f, df_r = self.vehicle.aero_model.calculate_downforce(
                300.0 / 3.6, "high_downforce", 0.05, 0.055
            )
            drag = self.vehicle.aero_model.calculate_total_drag(
                300.0 / 3.6, "high_downforce"
            )
            total_df = df_f + df_r
            return float(
                ((total_df - 22000.0) / 22000.0) ** 2
                + ((drag - 11000.0) / 11000.0) ** 2
            )

        x_best, f_best, ok, method = self._safe_optimize(objective, x0, bounds)
        modes["high_downforce"]["CL_front"] = float(x_best[0])
        modes["high_downforce"]["CL_rear"] = float(x_best[1])
        modes["high_downforce"]["CD_base"] = float(x_best[2])

        return CalibrationResult(
            parameters={
                "CL_front_high": float(x_best[0]),
                "CL_rear_high": float(x_best[1]),
                "CD_high": float(x_best[2]),
            },
            objective=float(f_best),
            method=method,
            success=ok,
        )

    def calibrate_powertrain_parameters(self) -> CalibrationResult:
        ice = self.vehicle.powertrain.ice
        mguk = self.vehicle.powertrain.mguk

        x0 = np.asarray([ice.max_power, mguk.max_motor_power], dtype=np.float64)
        bounds = [(340000.0, 390000.0), (340000.0, 390000.0)]

        def objective(x: np.ndarray) -> float:
            ice.max_power = float(x[0])
            mguk.max_motor_power = float(x[1])

            self.vehicle.powertrain.reset(110.0)
            self.vehicle.powertrain.battery.state.state_of_charge = 0.8
            self.vehicle.powertrain.battery.state.energy_j = (
                0.8 * self.vehicle.powertrain.battery.capacity
            )

            for _ in range(10):
                self.vehicle.powertrain.update(0.02)
                _, _, _, st = self.vehicle.powertrain.calculate_wheel_power(
                    throttle=1.0,
                    brake=0.0,
                    rpm=11000.0,
                    gear=6,
                    vehicle_speed=80.0,
                    deployment_strategy="max_power",
                    dt=0.02,
                )
            total_kw = float(st["total_power_w"]) / 1000.0
            split_err = abs(float(st["power_split_ice"]) - 0.5)
            return float(((total_kw - 746.0) / 746.0) ** 2 + split_err**2)

        x_best, f_best, ok, method = self._safe_optimize(objective, x0, bounds)
        ice.max_power = float(x_best[0])
        mguk.max_motor_power = float(x_best[1])

        return CalibrationResult(
            parameters={
                "ice_max_power_w": float(x_best[0]),
                "mguk_max_power_w": float(x_best[1]),
            },
            objective=float(f_best),
            method=method,
            success=ok,
        )

    def calibrate_all(self) -> dict[str, Any]:
        tire = self.calibrate_tire_parameters()
        aero = self.calibrate_aero_parameters()
        power = self.calibrate_powertrain_parameters()
        return {
            "tire": tire.__dict__,
            "aero": aero.__dict__,
            "powertrain": power.__dict__,
        }
