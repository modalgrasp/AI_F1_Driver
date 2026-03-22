#!/usr/bin/env python3
"""Aerodynamics model for F1 2026 active aero regulations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_AERO_CONFIG_PATH = Path("configs/aero_config.json")
EPSILON = 1e-6


@dataclass
class CoolingSystem:
    """Simple cooling drag model for brake ducts and radiators."""

    brake_duct_opening: float = 0.8
    radiator_opening: float = 0.8

    def get_cooling_drag(self, cooling_cfg: dict[str, float]) -> float:
        return self.brake_duct_opening * float(
            cooling_cfg["brake_duct_drag"]
        ) + self.radiator_opening * float(cooling_cfg["radiator_drag"])

    def adjust_for_conditions(self, temperature_c: float, speed_mps: float) -> None:
        thermal_demand = np.clip((temperature_c - 25.0) / 30.0, 0.0, 1.0)
        speed_relief = np.clip(speed_mps / 90.0, 0.0, 1.0)
        opening = np.clip(0.6 + 0.5 * thermal_demand - 0.25 * speed_relief, 0.35, 1.0)
        self.brake_duct_opening = float(opening)
        self.radiator_opening = float(opening)


class AerodynamicsModel:
    """Aerodynamics model for F1 2026 with active aero."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self._load_config(config)
        self.reference = self.config["reference"]
        self.modes = self.config["modes"]
        self.ground_effect_cfg = self.config["ground_effect"]
        self.dirty_air_cfg = self.config["dirty_air"]
        self.environmental_cfg = self.config["environmental"]
        self.cooling_cfg = self.config["cooling"]

        self.cooling = CoolingSystem()
        self.current_mode = "high_downforce"
        self.target_mode = "high_downforce"
        self.transition_progress = 1.0
        self.transition_time = float(self.modes[self.current_mode]["transition_time"])
        self.air_density = self.calculate_air_density(
            float(self.environmental_cfg["reference_temperature"]),
            float(self.environmental_cfg["reference_pressure"]),
            float(self.environmental_cfg["reference_humidity"]),
        )

    def _load_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return config
        if DEFAULT_AERO_CONFIG_PATH.exists():
            with DEFAULT_AERO_CONFIG_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        raise FileNotFoundError(
            f"Aerodynamics config not found at {DEFAULT_AERO_CONFIG_PATH.as_posix()}"
        )

    @staticmethod
    def _interp(a: float, b: float, t: float) -> float:
        return a + (b - a) * np.clip(t, 0.0, 1.0)

    def update_aero_mode(self, target_mode: str, dt: float) -> None:
        """Transition between aero modes with actuator lag."""
        if target_mode not in self.modes:
            raise ValueError("Aero mode must be 'low_drag' or 'high_downforce'.")

        if target_mode != self.target_mode:
            self.target_mode = target_mode
            self.transition_progress = 0.0
            self.transition_time = float(self.modes[target_mode]["transition_time"])

        if self.current_mode != self.target_mode:
            self.transition_progress += dt / max(self.transition_time, EPSILON)
            if self.transition_progress >= 1.0:
                self.current_mode = self.target_mode
                self.transition_progress = 1.0

    def get_lift_coefficients(
        self, aero_mode: str | None = None
    ) -> tuple[float, float]:
        """Return mode lift coefficients; interpolated during transitions."""
        if aero_mode is not None and aero_mode in self.modes:
            mode_cfg = self.modes[aero_mode]
            return float(mode_cfg["CL_front"]), float(mode_cfg["CL_rear"])

        if self.current_mode == self.target_mode:
            mode_cfg = self.modes[self.current_mode]
            return float(mode_cfg["CL_front"]), float(mode_cfg["CL_rear"])

        start_cfg = self.modes[self.current_mode]
        target_cfg = self.modes[self.target_mode]
        t = float(np.clip(self.transition_progress, 0.0, 1.0))
        cl_front = self._interp(
            float(start_cfg["CL_front"]), float(target_cfg["CL_front"]), t
        )
        cl_rear = self._interp(
            float(start_cfg["CL_rear"]), float(target_cfg["CL_rear"]), t
        )
        return cl_front, cl_rear

    def get_drag_coefficient(self, aero_mode: str | None = None) -> float:
        """Return base drag coefficient for active mode state."""
        if aero_mode is not None and aero_mode in self.modes:
            return float(self.modes[aero_mode]["CD_base"])

        if self.current_mode == self.target_mode:
            return float(self.modes[self.current_mode]["CD_base"])

        start_cd = float(self.modes[self.current_mode]["CD_base"])
        target_cd = float(self.modes[self.target_mode]["CD_base"])
        return self._interp(start_cd, target_cd, float(self.transition_progress))

    def set_aero_mode(self, mode: str) -> None:
        """Set target aero mode ('low_drag' or 'high_downforce')."""
        if mode not in self.modes:
            raise ValueError("Aero mode must be 'low_drag' or 'high_downforce'.")
        self.target_mode = mode
        if self.current_mode != mode:
            self.transition_progress = 0.0
            self.transition_time = float(self.modes[mode]["transition_time"])

    def update(self, dt: float) -> None:
        """Advance mode transitions by dt."""
        self.update_aero_mode(self.target_mode, dt)

    def ground_effect_multiplier(
        self, ride_height: float | np.ndarray
    ) -> float | np.ndarray:
        """Ride-height dependent ground-effect multiplier."""
        rh = np.asarray(ride_height, dtype=np.float64)
        optimal = float(self.ground_effect_cfg["optimal_ride_height"])
        min_height = float(self.ground_effect_cfg["min_ride_height"])
        sensitivity = float(self.ground_effect_cfg["sensitivity"])

        low_region = np.where(rh < min_height, 0.5, 1.0)
        below_opt = (
            1.0 + np.clip((optimal - rh) / max(optimal, EPSILON), 0.0, 1.0) * 0.5
        )
        above_opt = 1.0 - np.clip((rh - optimal) / 0.10, 0.0, 1.0) * sensitivity
        mult = np.where(
            rh < min_height, low_region, np.where(rh < optimal, below_opt, above_opt)
        )
        mult = np.clip(mult, 0.45, 1.55)
        if mult.ndim == 0:
            return float(mult)
        return mult

    def dirty_air_multiplier(self, distance_to_car_ahead: float | None) -> float:
        """Downforce multiplier in turbulent wake from a leading car."""
        if distance_to_car_ahead is None:
            return 1.0
        distance = float(distance_to_car_ahead)
        max_loss = float(self.dirty_air_cfg["max_downforce_loss"])
        peak_distance = float(self.dirty_air_cfg["peak_distance"])
        effect_range = float(self.dirty_air_cfg["effect_range"])
        if distance < 2.0:
            return 0.95
        if distance < effect_range:
            loss = max_loss * np.exp(-((distance - peak_distance) ** 2) / 10.0)
            return float(np.clip(1.0 - loss, 0.7, 1.0))
        return 1.0

    @staticmethod
    def drs_effect(drs_active: bool) -> dict[str, float]:
        if drs_active:
            return {
                "drag_multiplier": 0.85,
                "rear_downforce_multiplier": 0.92,
            }
        return {
            "drag_multiplier": 1.0,
            "rear_downforce_multiplier": 1.0,
        }

    @staticmethod
    def pitch_sensitivity(pitch_angle: float) -> tuple[float, float]:
        front_mult = np.clip(1.0 - 5.0 * pitch_angle, 0.7, 1.25)
        rear_mult = np.clip(1.0 + 3.0 * pitch_angle, 0.7, 1.25)
        return float(front_mult), float(rear_mult)

    @staticmethod
    def roll_sensitivity(roll_angle: float) -> float:
        return float(np.clip(1.0 - 0.5 * abs(roll_angle), 0.75, 1.0))

    def reynolds_number_effect(self, speed: float) -> float:
        rho = self.air_density
        visc = float(self.environmental_cfg["air_viscosity"])
        ref_length = float(self.reference["reference_length"])
        re = (rho * max(speed, 0.0) * ref_length) / max(visc, EPSILON)
        if re <= 0.0:
            return 1.0
        return float(np.clip(1.0 + 0.01 * np.log10(re / 1e7 + 1.0), 0.97, 1.03))

    @staticmethod
    def calculate_air_density(
        temperature: float, pressure: float, humidity: float
    ) -> float:
        """Air density from ideal gas law with humidity correction."""
        r_dry = 287.05
        r_vapor = 461.5
        t_kelvin = temperature + 273.15
        p_sat = 610.78 * np.exp(17.27 * temperature / (temperature + 237.3))
        p_vapor = np.clip(humidity, 0.0, 1.0) * p_sat
        p_dry = max(pressure - p_vapor, 1000.0)
        rho = p_dry / (r_dry * t_kelvin) + p_vapor / (r_vapor * t_kelvin)
        return float(np.clip(rho, 0.8, 1.4))

    def calculate_downforce(
        self,
        speed: float,
        aero_mode: str | None,
        ride_height_front: float,
        ride_height_rear: float,
        pitch: float = 0.0,
        roll: float = 0.0,
        distance_to_car_ahead: float | None = None,
        drs_active: bool = False,
    ) -> tuple[float, float]:
        """Calculate front and rear downforce components."""
        cl_front, cl_rear = self.get_lift_coefficients(aero_mode)
        q = 0.5 * self.air_density * max(speed, 0.0) ** 2
        reynolds_mult = self.reynolds_number_effect(speed)
        dirty_mult = self.dirty_air_multiplier(distance_to_car_ahead)

        ge_front = self.ground_effect_multiplier(ride_height_front)
        ge_rear = self.ground_effect_multiplier(ride_height_rear)
        pitch_front_mult, pitch_rear_mult = self.pitch_sensitivity(pitch)
        roll_mult = self.roll_sensitivity(roll)
        drs = self.drs_effect(drs_active)

        front_area = float(self.reference["front_area"])
        rear_area = float(self.reference["rear_area"])
        downforce_front = q * front_area * abs(cl_front) * ge_front
        downforce_rear = q * rear_area * abs(cl_rear) * ge_rear

        downforce_front *= reynolds_mult * dirty_mult * pitch_front_mult * roll_mult
        downforce_rear *= (
            reynolds_mult
            * (0.92 + 0.08 * dirty_mult)
            * pitch_rear_mult
            * roll_mult
            * drs["rear_downforce_multiplier"]
        )
        return float(downforce_front), float(downforce_rear)

    def calculate_total_drag(
        self,
        speed: float,
        aero_mode: str | None = None,
        cooling_setting: float | None = None,
        drs_active: bool = False,
    ) -> float:
        """Total drag including wings/body and cooling effects."""
        q = 0.5 * self.air_density * max(speed, 0.0) ** 2
        cd_base = self.get_drag_coefficient(aero_mode)
        if cooling_setting is None:
            cd_cooling = self.cooling.get_cooling_drag(self.cooling_cfg)
        else:
            cd_cooling = np.clip(cooling_setting, 0.0, 1.0) * float(
                self.cooling_cfg["radiator_drag"]
            )

        cd_wheels = 0.10
        cd_total = cd_base + cd_cooling + cd_wheels
        drag = q * float(self.reference["frontal_area"]) * cd_total
        drag *= self.drs_effect(drs_active)["drag_multiplier"]
        return float(max(drag, 0.0))

    def calculate_cop(self, downforce_front: float, downforce_rear: float) -> float:
        """Center of pressure measured from front axle."""
        total = downforce_front + downforce_rear
        wheelbase = float(self.reference["wheelbase"])
        if total <= EPSILON:
            return wheelbase * 0.5
        return float((downforce_rear * wheelbase) / total)

    def calculate_forces(
        self,
        speed: float,
        aero_mode: str,
        ride_height_front: float,
        ride_height_rear: float,
        pitch: float,
        roll: float,
        yaw: float,
        distance_to_car_ahead: float | None = None,
        drs_active: bool = False,
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate aerodynamic forces and moments."""
        if aero_mode != self.target_mode:
            self.set_aero_mode(aero_mode)

        downforce_front, downforce_rear = self.calculate_downforce(
            speed=speed,
            aero_mode=None,
            ride_height_front=ride_height_front,
            ride_height_rear=ride_height_rear,
            pitch=pitch,
            roll=roll,
            distance_to_car_ahead=distance_to_car_ahead,
            drs_active=drs_active,
        )
        drag = self.calculate_total_drag(
            speed=speed, aero_mode=None, drs_active=drs_active
        )

        wheelbase = float(self.reference["wheelbase"])
        cop = self.calculate_cop(downforce_front, downforce_rear)
        total_downforce = downforce_front + downforce_rear

        pitch_moment = (cop - wheelbase * 0.5) * total_downforce
        roll_moment = roll * 0.05 * total_downforce
        yaw_moment = -yaw * 0.02 * (drag + total_downforce)

        return (
            float(downforce_front),
            float(downforce_rear),
            float(drag),
            float(pitch_moment),
            float(roll_moment),
            float(yaw_moment),
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "target_mode": self.target_mode,
            "transition_progress": self.transition_progress,
            "air_density": self.air_density,
            "cooling": {
                "brake_duct_opening": self.cooling.brake_duct_opening,
                "radiator_opening": self.cooling.radiator_opening,
            },
        }

    def reset(self) -> None:
        self.current_mode = "high_downforce"
        self.target_mode = "high_downforce"
        self.transition_progress = 1.0
        self.transition_time = float(self.modes[self.current_mode]["transition_time"])
