#!/usr/bin/env python3
"""Pacejka Magic Formula tire model for F1 2026 simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

EPSILON = 1e-6
PSI_TO_BAR = 0.0689476
DEFAULT_TIRE_CONFIG_PATH = Path("configs/tire_config.json")


@dataclass
class TireState:
    """Mutable tire state used by the thermal and degradation model."""

    wear_percentage: float = 0.0
    surface_temp: float = 85.0
    core_temp: float = 80.0
    pressure: float = 1.38
    compound: str = "medium"
    inner_temp: float = 85.0
    middle_temp: float = 85.0
    outer_temp: float = 85.0


class TireModel:
    """Pacejka Magic Formula tire model for F1 2026.

    Implements pure and combined slip forces, temperature effects, degradation,
    pressure influence, and self-aligning moment.
    """

    def __init__(
        self, compound: str = "medium", config: dict[str, Any] | None = None
    ) -> None:
        self.config = self._load_config(config)
        self.environmental = self.config.get("environmental", {})
        self._compounds = self.config["compounds"]
        self.compound = ""
        self.params: dict[str, Any] = {}
        self.state = TireState()
        self._extreme_slip_ratio_warned = False
        self._extreme_slip_angle_warned = False
        self.reset(compound=compound)

    def _load_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return config
        if DEFAULT_TIRE_CONFIG_PATH.exists():
            with DEFAULT_TIRE_CONFIG_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        raise FileNotFoundError(
            f"Tire config not found at {DEFAULT_TIRE_CONFIG_PATH.as_posix()}"
        )

    @staticmethod
    def _as_array(value: float | np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _return(value: np.ndarray) -> float | np.ndarray:
        if value.ndim == 0:
            return float(value)
        return value

    @staticmethod
    def calculate_slip_angle(
        vx: float | np.ndarray, vy: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute slip angle using atan2 for quadrant-safe behavior."""
        vx_arr = np.asarray(vx, dtype=np.float64)
        vy_arr = np.asarray(vy, dtype=np.float64)
        safe_vx = np.where(np.abs(vx_arr) < EPSILON, np.sign(vx_arr) * EPSILON, vx_arr)
        alpha = np.arctan2(vy_arr, safe_vx)
        if alpha.ndim == 0:
            return float(alpha)
        return alpha

    def calculate_slip_ratio(
        self, wheel_speed: float | np.ndarray, vehicle_speed: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute longitudinal slip ratio with lock/spin-safe denominator."""
        omega = self._as_array(wheel_speed)
        vx = self._as_array(vehicle_speed)
        radius = float(self.params["physical"]["radius"])

        wheel_linear = omega * radius
        denom = np.maximum(np.abs(vx), 1.0)
        kappa = (wheel_linear - vx) / denom
        kappa = np.clip(kappa, -1.2, 2.5)
        return self._return(kappa)

    @staticmethod
    def _magic_formula(
        slip: np.ndarray,
        b_coeff: float | np.ndarray,
        c_coeff: float,
        d_coeff: np.ndarray,
        e_coeff: float,
    ) -> np.ndarray:
        b_x = b_coeff * slip
        atan_bx = np.arctan(b_x)
        term = b_x - e_coeff * (b_x - atan_bx)
        return d_coeff * np.sin(c_coeff * np.arctan(term))

    def _effective_mu(
        self,
        normal_load: np.ndarray,
        nominal_mu: float,
        load_sensitivity: float,
        wear: np.ndarray,
        temperature: np.ndarray,
        pressure_bar: np.ndarray,
    ) -> np.ndarray:
        fz_nom = float(self.params["normal_load_nominal"])
        delta = (normal_load - fz_nom) / fz_nom
        mu_load = nominal_mu * (1.0 - load_sensitivity * delta)
        grip_mult = self._grip_multiplier_array(temperature, wear, pressure_bar)
        mu = np.clip(mu_load * grip_mult, 0.4, 3.0)
        return mu

    def _camber_multiplier(self, camber_angle_rad: np.ndarray) -> np.ndarray:
        camber = self.params["camber"]
        optimal = np.deg2rad(float(camber["optimal_deg"]))
        k_camber = float(camber["sensitivity"])
        delta = camber_angle_rad - optimal
        mult = 1.0 - k_camber * delta * delta
        return np.clip(mult, 0.75, 1.05)

    def temperature_grip_multiplier(
        self, temperature_c: float | np.ndarray
    ) -> float | np.ndarray:
        """Gaussian-like grip curve around each compound's operating window."""
        temperature = self._as_array(temperature_c)
        temp_cfg = self.params["temperature"]
        temp_opt = float(temp_cfg["optimal"])
        temp_range = max(float(temp_cfg["range"]), EPSILON)
        deviation = np.abs(temperature - temp_opt)
        multiplier = np.exp(-((deviation / temp_range) ** 2))
        multiplier = np.clip(multiplier, 0.5, 1.02)
        return self._return(multiplier)

    def _wear_multiplier(self, wear: np.ndarray) -> np.ndarray:
        deg = self.params["degradation"]
        perf_loss = float(deg["performance_loss_coef"])
        cliff_threshold = float(deg["cliff_wear_threshold"])
        cliff_severity = float(deg["cliff_severity"])

        base = 1.0 - perf_loss * wear
        cliff_input = np.maximum(wear - cliff_threshold, 0.0) / max(
            100.0 - cliff_threshold, EPSILON
        )
        cliff = 1.0 - cliff_severity * cliff_input * cliff_input
        return np.clip(base * cliff, 0.3, 1.0)

    def _pressure_multiplier(self, pressure_bar: np.ndarray) -> np.ndarray:
        physical = self.params["physical"]
        optimal_bar = float(physical["optimal_pressure_psi"]) * PSI_TO_BAR
        delta = pressure_bar - optimal_bar
        mult = 1.0 - 0.12 * (delta / max(optimal_bar, EPSILON)) ** 2
        return np.clip(mult, 0.8, 1.02)

    def _graining_blistering_multiplier(self, temperature: np.ndarray) -> np.ndarray:
        temp_cfg = self.params["temperature"]
        min_op = float(temp_cfg["min_operating"])
        max_op = float(temp_cfg["max_operating"])
        graining_zone = np.clip((min_op - temperature) / 20.0, 0.0, 1.0)
        blister_zone = np.clip((temperature - max_op) / 20.0, 0.0, 1.0)
        return np.clip(1.0 - 0.2 * graining_zone - 0.3 * blister_zone, 0.55, 1.0)

    def _grip_multiplier_array(
        self,
        temperature: np.ndarray,
        wear: np.ndarray,
        pressure_bar: np.ndarray,
    ) -> np.ndarray:
        temp_mult = self._as_array(self.temperature_grip_multiplier(temperature))
        wear_mult = self._wear_multiplier(wear)
        pressure_mult = self._pressure_multiplier(pressure_bar)
        damage_mult = self._graining_blistering_multiplier(temperature)
        return np.clip(temp_mult * wear_mult * pressure_mult * damage_mult, 0.2, 1.05)

    def get_grip_multiplier(
        self, temperature: float | np.ndarray, wear: float | np.ndarray
    ) -> float | np.ndarray:
        """Calculate total grip multiplier from temperature and wear."""
        temperature_arr = self._as_array(temperature)
        wear_arr = self._as_array(wear)
        pressure = np.full_like(temperature_arr, self.state.pressure, dtype=np.float64)
        mult = self._grip_multiplier_array(temperature_arr, wear_arr, pressure)
        return self._return(mult)

    def calculate_lateral_force(
        self,
        slip_angle: float | np.ndarray,
        normal_load: float | np.ndarray,
        temp: float | np.ndarray,
        wear: float | np.ndarray,
        camber_angle: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """Pure lateral force from Magic Formula."""
        alpha = np.clip(self._as_array(slip_angle), -np.deg2rad(30.0), np.deg2rad(30.0))
        fz = np.maximum(self._as_array(normal_load), 50.0)
        temperature = self._as_array(temp)
        wear_arr = self._as_array(wear)
        camber = self._as_array(camber_angle)
        pressure = np.full_like(alpha, self.state.pressure, dtype=np.float64)

        lat = self.params["lateral_force"]
        mu = self._effective_mu(
            fz,
            nominal_mu=float(lat["D_nominal"]),
            load_sensitivity=float(lat["load_sensitivity"]),
            wear=wear_arr,
            temperature=temperature,
            pressure_bar=pressure,
        )
        d_coeff = mu * fz * self._camber_multiplier(camber)

        force = self._magic_formula(
            alpha,
            b_coeff=float(lat["B"]),
            c_coeff=float(lat["C"]),
            d_coeff=d_coeff,
            e_coeff=float(lat["E"]),
        )
        return self._return(force)

    def calculate_longitudinal_force(
        self,
        slip_ratio: float | np.ndarray,
        normal_load: float | np.ndarray,
        temp: float | np.ndarray,
        wear: float | np.ndarray,
    ) -> float | np.ndarray:
        """Pure longitudinal force from Magic Formula."""
        kappa = np.clip(self._as_array(slip_ratio), -1.2, 2.5)
        fz = np.maximum(self._as_array(normal_load), 50.0)
        temperature = self._as_array(temp)
        wear_arr = self._as_array(wear)
        pressure = np.full_like(kappa, self.state.pressure, dtype=np.float64)

        long_cfg = self.params["longitudinal_force"]
        braking_mask = kappa < 0.0

        b_coeff = np.where(
            braking_mask, float(long_cfg["B_brake"]), float(long_cfg["B_accel"])
        )
        d_nominal = np.where(
            braking_mask,
            float(long_cfg["D_nominal_brake"]),
            float(long_cfg["D_nominal_accel"]),
        )
        mu = (
            self._effective_mu(
                fz,
                nominal_mu=1.0,
                load_sensitivity=float(long_cfg["load_sensitivity"]),
                wear=wear_arr,
                temperature=temperature,
                pressure_bar=pressure,
            )
            * d_nominal
        )
        d_coeff = mu * fz

        force = self._magic_formula(
            kappa,
            b_coeff=b_coeff,
            c_coeff=float(long_cfg["C"]),
            d_coeff=d_coeff,
            e_coeff=float(long_cfg["E"]),
        )
        return self._return(force)

    def combined_force_scaling(
        self, slip_angle: float | np.ndarray, slip_ratio: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Calculate scaling factors for combined slip."""
        alpha = np.abs(self._as_array(slip_angle))
        kappa = np.abs(self._as_array(slip_ratio))
        comb_cfg = self.params["combined_slip"]
        bxa = float(comb_cfg["B_xalpha"])
        byk = float(comb_cfg["B_ykappa"])
        cxa = float(comb_cfg["C_xalpha"])
        cyk = float(comb_cfg["C_ykappa"])

        gx_alpha = np.cos(cxa * np.arctan(bxa * alpha))
        gy_kappa = np.cos(cyk * np.arctan(byk * kappa))
        scale = float(comb_cfg.get("scaling_factor", 1.0))
        gx_alpha = np.clip(gx_alpha * scale, 0.1, 1.0)
        gy_kappa = np.clip(gy_kappa * scale, 0.1, 1.0)
        return self._return(gx_alpha), self._return(gy_kappa)

    def calculate_combined_forces(
        self,
        slip_angle: float | np.ndarray,
        slip_ratio: float | np.ndarray,
        normal_load: float | np.ndarray,
        temp: float | np.ndarray,
        wear: float | np.ndarray,
        camber_angle: float | np.ndarray = 0.0,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Combined slip force computation with friction ellipse enforcement."""
        fx_pure = self._as_array(
            self.calculate_longitudinal_force(slip_ratio, normal_load, temp, wear)
        )
        fy_pure = self._as_array(
            self.calculate_lateral_force(
                slip_angle, normal_load, temp, wear, camber_angle
            )
        )
        gx_alpha, gy_kappa = self.combined_force_scaling(slip_angle, slip_ratio)
        fx = fx_pure * self._as_array(gx_alpha)
        fy = fy_pure * self._as_array(gy_kappa)

        # Enforce friction ellipse for physically plausible combined demand.
        ellipse = np.sqrt(
            (fx / np.maximum(np.abs(fx_pure), EPSILON)) ** 2
            + (fy / np.maximum(np.abs(fy_pure), EPSILON)) ** 2
        )
        safe_ellipse = np.maximum(ellipse, EPSILON)
        limiter = np.where(ellipse > 1.0, 1.0 / safe_ellipse, 1.0)
        fx *= limiter
        fy *= limiter
        return self._return(fx), self._return(fy)

    def calculate_rolling_resistance(
        self,
        normal_load: float | np.ndarray,
        vehicle_speed: float | np.ndarray,
        temperature: float | np.ndarray,
        pressure_bar: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Rolling resistance with temperature and pressure dependency."""
        fz = np.maximum(self._as_array(normal_load), 0.0)
        vx = self._as_array(vehicle_speed)
        temp = self._as_array(temperature)
        if pressure_bar is None:
            pressure = np.full_like(temp, self.state.pressure)
        else:
            pressure = self._as_array(pressure_bar)

        crr_base = float(self.params["rolling_resistance"]["C_rr_base"])
        temp_opt = float(self.params["temperature"]["optimal"])
        temp_factor = 1.0 + 0.003 * np.abs(temp - temp_opt)
        pressure_opt = (
            float(self.params["physical"]["optimal_pressure_psi"]) * PSI_TO_BAR
        )
        pressure_factor = 1.0 + 0.05 * np.abs(pressure - pressure_opt) / max(
            pressure_opt, EPSILON
        )
        crr = crr_base * temp_factor * pressure_factor

        sign = np.sign(vx)
        sign = np.where(np.abs(vx) < EPSILON, 0.0, sign)
        f_roll = crr * fz * sign
        return self._return(f_roll)

    def calculate_self_aligning_torque(
        self,
        lateral_force: float | np.ndarray,
        slip_angle: float | np.ndarray,
        normal_load: float | np.ndarray,
    ) -> float | np.ndarray:
        """Self-aligning torque from pneumatic trail model."""
        fy = self._as_array(lateral_force)
        alpha = np.abs(self._as_array(slip_angle))
        fz = np.maximum(self._as_array(normal_load), 50.0)

        trail_cfg = self.params["pneumatic_trail"]
        trail0 = float(trail_cfg["trail_nominal"])
        decay = float(trail_cfg["decay_rate"])
        fz_nom = float(self.params["normal_load_nominal"])

        trail = trail0 * np.exp(-decay * alpha) * np.clip(fz / fz_nom, 0.7, 1.2)
        mz = -trail * fy
        return self._return(mz)

    def calculate_forces(
        self,
        slip_angle: float | np.ndarray,
        slip_ratio: float | np.ndarray,
        normal_load: float | np.ndarray,
        temperature: float | np.ndarray,
        wear: float | np.ndarray,
        wheel_speed: float | np.ndarray,
        vehicle_speed: float | np.ndarray,
        camber_angle: float | np.ndarray = 0.0,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Calculate combined tire forces and self-aligning torque."""
        fx, fy = self.calculate_combined_forces(
            slip_angle=slip_angle,
            slip_ratio=slip_ratio,
            normal_load=normal_load,
            temp=temperature,
            wear=wear,
            camber_angle=camber_angle,
        )
        f_roll = self.calculate_rolling_resistance(
            normal_load, vehicle_speed, temperature
        )
        fx_total = self._as_array(fx) - self._as_array(f_roll)
        mz = self.calculate_self_aligning_torque(fy, slip_angle, normal_load)

        if (
            np.any(np.abs(self._as_array(slip_ratio)) > 1.5)
            and not self._extreme_slip_ratio_warned
        ):
            LOGGER.warning("Extreme slip ratio detected; tire likely wheel spin/lock.")
            self._extreme_slip_ratio_warned = True
        if (
            np.any(np.abs(self._as_array(slip_angle)) > np.deg2rad(20.0))
            and not self._extreme_slip_angle_warned
        ):
            LOGGER.warning(
                "Extreme slip angle detected; tire likely beyond optimal range."
            )
            self._extreme_slip_angle_warned = True

        return self._return(fx_total), fy, mz

    def update_temperature(
        self,
        forces: dict[str, float | np.ndarray],
        speeds: dict[str, float | np.ndarray],
        ambient_temp: float,
        dt: float,
    ) -> dict[str, float]:
        """Update surface/core temperatures using lumped thermal model."""
        fx = self._as_array(forces.get("Fx", 0.0))
        fy = self._as_array(forces.get("Fy", 0.0))
        vx_slip = self._as_array(speeds.get("vx_slip", 0.0))
        vy_slip = self._as_array(speeds.get("vy_slip", 0.0))
        track_temp = float(speeds.get("track_temp", ambient_temp))
        camber = float(speeds.get("camber", 0.0))

        q_slip = np.abs(fx * vx_slip) + np.abs(fy * vy_slip)
        thermal = self.params["thermal"]
        h_conv = float(thermal["h_convection"])
        k_core = float(thermal["k_surface_to_core"])
        m_surface = float(thermal["mass_surface"])
        m_core = float(thermal["mass_core"])
        cp = float(thermal["cp"])

        t_surface = self._as_array(self.state.surface_temp)
        t_core = self._as_array(self.state.core_temp)

        q_convection = h_conv * (t_surface - ambient_temp)
        q_track = 0.5 * h_conv * (track_temp - t_surface)
        q_conduction = k_core * (t_surface - t_core)

        t_surface_dot = (q_slip + q_track - q_convection - q_conduction) / (
            m_surface * cp
        )
        t_core_dot = q_conduction / (m_core * cp)

        self.state.surface_temp = float(
            np.clip(t_surface + t_surface_dot * dt, -20.0, 220.0)
        )
        self.state.core_temp = float(np.clip(t_core + t_core_dot * dt, -20.0, 200.0))

        # 3-zone profile: camber moves load/heat to inner shoulder for negative camber.
        camber_effect = np.clip(-camber, -0.2, 0.2)
        slip_skew = np.clip(float(np.mean(np.abs(vy_slip))) * 0.05, 0.0, 8.0)
        base = self.state.surface_temp
        self.state.inner_temp = base * (1.0 + 0.08 * camber_effect) + slip_skew
        self.state.middle_temp = base
        self.state.outer_temp = base * (1.0 - 0.08 * camber_effect) - slip_skew * 0.5

        # Hot pressure approximation via ideal-gas scaling from 20C reference.
        t_ref_k = 293.15
        t_hot_k = self.state.core_temp + 273.15
        p_cold_bar = float(self.params["physical"]["cold_pressure_psi"]) * PSI_TO_BAR
        self.state.pressure = float(np.clip(p_cold_bar * (t_hot_k / t_ref_k), 1.0, 2.4))

        return {
            "surface_temp": self.state.surface_temp,
            "core_temp": self.state.core_temp,
            "inner_temp": self.state.inner_temp,
            "middle_temp": self.state.middle_temp,
            "outer_temp": self.state.outer_temp,
            "pressure": self.state.pressure,
        }

    def update_wear(
        self,
        slip_angle: float | np.ndarray,
        slip_ratio: float | np.ndarray,
        temperature: float,
        dt: float,
    ) -> float:
        """Update monotonic tire wear percentage."""
        alpha = np.abs(self._as_array(slip_angle))
        kappa = np.abs(self._as_array(slip_ratio))
        degradation = self.params["degradation"]
        wear_base = float(degradation["wear_rate_base"])
        n_lat = float(degradation["n_lat"])
        n_long = float(degradation["n_long"])

        # Temperature accelerates wear outside the center of operating window.
        temp_opt = float(self.params["temperature"]["optimal"])
        temp_mult = 1.0 + 0.02 * np.abs(temperature - temp_opt)

        wear_rate = wear_base * ((alpha**n_lat) + (kappa**n_long)) * temp_mult
        delta_wear = float(np.mean(wear_rate) * dt)
        self.state.wear_percentage = float(
            np.clip(self.state.wear_percentage + delta_wear, 0.0, 100.0)
        )
        return self.state.wear_percentage

    @staticmethod
    def calculate_longitudinal_load_transfer(
        mass: float,
        acceleration_longitudinal: float,
        cg_height: float,
        wheelbase: float,
    ) -> float:
        """Return front-to-rear load transfer magnitude (N)."""
        if wheelbase <= EPSILON:
            return 0.0
        return mass * acceleration_longitudinal * cg_height / wheelbase

    def get_state(self) -> dict[str, Any]:
        """Return current tire state as dictionary."""
        return asdict(self.state)

    def set_state(self, state_dict: dict[str, Any]) -> None:
        """Set tire state from dictionary."""
        valid = TireState().__dict__.keys()
        merged = self.get_state()
        for key, value in state_dict.items():
            if key in valid:
                merged[key] = value
        self.state = TireState(**merged)

    def reset(self, compound: str = "medium") -> None:
        """Reset tire state and switch compound parameters."""
        if compound not in self._compounds:
            raise ValueError(f"Unknown compound '{compound}'.")
        self.compound = compound
        self.params = self._compounds[compound]
        temp_opt = float(self.params["temperature"]["optimal"])
        cold_pressure_bar = (
            float(self.params["physical"]["cold_pressure_psi"]) * PSI_TO_BAR
        )
        self.state = TireState(
            wear_percentage=0.0,
            surface_temp=temp_opt - 12.0,
            core_temp=temp_opt - 18.0,
            pressure=cold_pressure_bar,
            compound=compound,
            inner_temp=temp_opt - 12.0,
            middle_temp=temp_opt - 12.0,
            outer_temp=temp_opt - 12.0,
        )
        self._extreme_slip_ratio_warned = False
        self._extreme_slip_angle_warned = False
