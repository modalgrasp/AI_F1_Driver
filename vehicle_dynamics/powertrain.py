#!/usr/bin/env python3
"""Hybrid powertrain model for F1 2026 regulations."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

EPSILON = 1e-9
DEFAULT_POWERTRAIN_CONFIG_PATH = Path("configs/powertrain_config.json")


@dataclass
class ICEState:
    rpm: float = 5000.0
    boost_pressure_bar: float = 0.1
    power_w: float = 0.0
    torque_nm: float = 0.0
    fuel_flow_kg_s: float = 0.0


class ICEModel:
    """Internal combustion engine model (1.6L turbo V6)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.displacement = float(config["displacement"])
        self.cylinders = int(config["cylinders"])
        self.max_rpm = float(config["max_rpm"])
        self.idle_rpm = float(config["idle_rpm"])
        self.max_power = float(config["max_power"])
        self.max_torque = float(config["max_torque"])
        self.peak_rpm = float(config["peak_rpm"])
        self.redline = float(config["redline"])
        self.bsfc_min = float(config["bsfc_min"])
        self.max_fuel_flow = 100.0 / 3600.0
        self.max_boost = float(config.get("max_boost_bar", 3.5))
        self.boost_tau = float(config.get("turbo_tau", 0.3))
        self.boost_pressure = float(config.get("initial_boost_bar", 0.1))
        self.state = ICEState(rpm=self.idle_rpm, boost_pressure_bar=self.boost_pressure)

    def get_bsfc(self, rpm: float, throttle: float) -> float:
        """Brake specific fuel consumption (kg/kWh)."""
        rpm_norm = np.clip(rpm / max(self.peak_rpm, 1.0), 0.5, 1.6)
        throttle_penalty = (1.0 - np.clip(throttle, 0.0, 1.0)) * 0.08
        rpm_penalty = (rpm_norm - 1.0) ** 2 * 0.06
        return float(
            np.clip(self.bsfc_min + throttle_penalty + rpm_penalty, 0.20, 0.42)
        )

    def get_target_boost(self, rpm: float, throttle: float) -> float:
        rpm_factor = np.clip(
            (rpm - self.idle_rpm) / max(self.peak_rpm - self.idle_rpm, 1.0), 0.0, 1.0
        )
        return float(self.max_boost * throttle * (0.55 + 0.45 * rpm_factor))

    def update_turbo_boost(self, rpm: float, throttle: float, dt: float) -> float:
        target_boost = self.get_target_boost(rpm, throttle)
        alpha = np.clip(dt / max(self.boost_tau, EPSILON), 0.0, 1.0)
        self.boost_pressure += (target_boost - self.boost_pressure) * alpha
        self.boost_pressure = float(np.clip(self.boost_pressure, 0.0, self.max_boost))
        return 1.0 + (self.boost_pressure / max(self.max_boost, EPSILON)) * 0.5

    @staticmethod
    def calculate_engine_braking(rpm: float, throttle: float) -> float:
        if throttle < 0.05:
            return float(-50.0 - (rpm / 15000.0) * 100.0)
        return 0.0

    def calculate_ice_power(
        self, rpm: float, throttle: float, dt: float = 0.01
    ) -> tuple[float, float, float]:
        """Compute ICE power, torque, and fuel flow."""
        rpm = float(np.clip(rpm, self.idle_rpm, self.max_rpm))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        self.state.rpm = rpm

        normalized_rpm = rpm / max(self.max_rpm, 1.0)
        shape = (
            -2.5 * normalized_rpm**3 + 3.8 * normalized_rpm**2 + 0.2 * normalized_rpm
        )
        torque_base = self.max_torque * max(shape, 0.0)

        boost_mult = self.update_turbo_boost(rpm, throttle, dt)
        torque = torque_base * throttle * boost_mult
        torque += self.calculate_engine_braking(rpm, throttle)

        omega = rpm * 2.0 * np.pi / 60.0
        power = max(torque * omega, 0.0)
        power = min(power, self.max_power)

        bsfc = self.get_bsfc(rpm, throttle)
        fuel_flow = min((power / 3600.0) * bsfc, self.max_fuel_flow)

        self.state.boost_pressure_bar = self.boost_pressure
        self.state.power_w = power
        self.state.torque_nm = torque
        self.state.fuel_flow_kg_s = fuel_flow
        return power, torque, fuel_flow

    def get_exhaust_energy(self, rpm: float, throttle: float = 1.0) -> float:
        """Approximate recoverable exhaust power."""
        rpm_factor = np.clip(rpm / max(self.max_rpm, 1.0), 0.0, 1.0)
        return float(self.max_power * 0.45 * rpm_factor * np.clip(throttle, 0.0, 1.0))


@dataclass
class TransmissionState:
    current_gear: int = 1
    in_shift: bool = False
    shift_timer: float = 0.0


class Transmission:
    """Eight-speed sequential gearbox with shift latency."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.num_gears = int(config["num_gears"])
        ratios = list(config["gear_ratios"])
        self.gear_ratios = {idx + 1: float(r) for idx, r in enumerate(ratios)}
        self.final_drive = float(config["final_drive"])
        self.efficiency = float(config["efficiency"])
        self.shift_time = float(config["shift_time"])
        self.state = TransmissionState()

    @property
    def current_gear(self) -> int:
        return self.state.current_gear

    def calculate_wheel_torque(
        self, engine_torque: float, gear: int | None = None
    ) -> float:
        selected_gear = self.current_gear if gear is None else int(gear)
        total_ratio = self.gear_ratios[selected_gear] * self.final_drive
        if self.state.in_shift:
            return 0.0
        return float(engine_torque * total_ratio * self.efficiency)

    def calculate_engine_rpm(
        self, wheel_speed_rad_s: float, gear: int | None = None
    ) -> float:
        selected_gear = self.current_gear if gear is None else int(gear)
        total_ratio = self.gear_ratios[selected_gear] * self.final_drive
        return float(wheel_speed_rad_s * total_ratio * 60.0 / (2.0 * np.pi))

    def shift_up(self) -> None:
        if self.current_gear < self.num_gears and not self.state.in_shift:
            self.state.current_gear += 1
            self.state.in_shift = True
            self.state.shift_timer = self.shift_time

    def shift_down(self) -> None:
        if self.current_gear > 1 and not self.state.in_shift:
            self.state.current_gear -= 1
            self.state.in_shift = True
            self.state.shift_timer = self.shift_time

    def update(self, dt: float) -> None:
        if self.state.in_shift:
            self.state.shift_timer -= dt
            if self.state.shift_timer <= 0.0:
                self.state.in_shift = False
                self.state.shift_timer = 0.0

    def reset(self) -> None:
        self.state = TransmissionState()


@dataclass
class MGUKState:
    power_w: float = 0.0
    torque_nm: float = 0.0
    battery_flow_w: float = 0.0


class MGUKModel:
    """Motor Generator Unit - Kinetic (drive + regen)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_motor_power = float(config["max_motor_power"])
        self.max_generator_power = float(config["max_generator_power"])
        self.efficiency_motor = float(config["efficiency_motor"])
        self.efficiency_generator = float(config["efficiency_generator"])
        self.max_torque = float(config["max_torque"])
        self.base_speed = float(config.get("base_speed_rpm", 7000.0))
        self.state = MGUKState()

    def get_max_torque(self, rpm: float) -> float:
        if rpm < self.base_speed:
            return self.max_torque
        return float(self.max_torque * self.base_speed / max(rpm, 1.0))

    def get_max_harvest_power(self, rpm: float) -> float:
        rpm_factor = np.clip((rpm - 3000.0) / 9000.0, 0.2, 1.0)
        return float(self.max_generator_power * rpm_factor)

    def calculate_motor_power(
        self, deployment_rate: float, rpm: float
    ) -> tuple[float, float, float]:
        deployment_rate = float(np.clip(deployment_rate, 0.0, 1.0))
        target_power = deployment_rate * self.max_motor_power
        max_torque_at_rpm = self.get_max_torque(rpm)
        omega = rpm * 2.0 * np.pi / 60.0
        max_power_at_rpm = max_torque_at_rpm * omega
        power_out = min(target_power, max_power_at_rpm)
        torque = 0.0 if rpm <= 100.0 else power_out / max(omega, EPSILON)
        battery_draw = power_out / max(self.efficiency_motor, EPSILON)
        self.state = MGUKState(
            power_w=power_out, torque_nm=torque, battery_flow_w=-battery_draw
        )
        return power_out, torque, battery_draw

    def calculate_generator_power(
        self, harvest_rate: float, rpm: float, braking: bool
    ) -> tuple[float, float]:
        harvest_rate = float(np.clip(harvest_rate, 0.0, 1.0))
        if not braking and harvest_rate < 0.1:
            self.state = MGUKState(power_w=0.0, torque_nm=0.0, battery_flow_w=0.0)
            return 0.0, 0.0

        target_power = harvest_rate * self.max_generator_power
        harvest_power = min(target_power, self.get_max_harvest_power(rpm))
        omega = rpm * 2.0 * np.pi / 60.0
        regen_torque = 0.0 if rpm <= 100.0 else -harvest_power / max(omega, EPSILON)
        battery_charge = harvest_power * self.efficiency_generator
        self.state = MGUKState(
            power_w=-harvest_power,
            torque_nm=regen_torque,
            battery_flow_w=battery_charge,
        )
        return regen_torque, battery_charge


class MGUHModel:
    """Motor Generator Unit - Heat (optional in 2026 config)."""

    def __init__(self, max_power: float = 120000.0, efficiency: float = 0.85) -> None:
        self.max_power = float(max_power)
        self.efficiency = float(efficiency)

    def calculate_harvest(self, exhaust_energy: float, rpm: float) -> float:
        rpm_factor = np.clip(rpm / 15000.0, 0.3, 1.0)
        return float(min(exhaust_energy * self.efficiency * rpm_factor, self.max_power))

    def calculate_turbo_assist(
        self, target_boost: float, current_boost: float
    ) -> float:
        boost_deficit = max(target_boost - current_boost, 0.0)
        return float(min(boost_deficit * 50000.0, self.max_power))


@dataclass
class BatteryState:
    state_of_charge: float
    energy_j: float
    temperature_c: float


class EnergyStore:
    """High-power battery and thermal state."""

    def __init__(self, config: dict[str, Any]) -> None:
        capacity_kwh = float(config["capacity_kwh"])
        self.capacity = capacity_kwh * 3600.0 * 1000.0
        self.max_charge_power = float(config["max_charge_rate"])
        self.max_discharge_power = float(config["max_discharge_rate"])
        self.min_soc = float(config["min_soc"])
        self.initial_soc = float(config["initial_soc"])
        temp_range = config["optimal_temp_range"]
        self.optimal_temp_min = float(temp_range[0])
        self.optimal_temp_max = float(temp_range[1])
        self.state = BatteryState(
            state_of_charge=self.initial_soc,
            energy_j=self.capacity * self.initial_soc,
            temperature_c=float(config.get("initial_temp_c", 25.0)),
        )

    @property
    def soc(self) -> float:
        return self.state.state_of_charge

    def update_temperature(self, heat_input_w: float, dt: float) -> None:
        mass = 50.0
        cp = 1000.0
        ambient = 40.0
        cooling = (self.state.temperature_c - ambient) * 100.0
        delta_t = (heat_input_w - cooling) / (mass * cp) * dt
        self.state.temperature_c += delta_t

    def get_performance_factor(self) -> float:
        temp = self.state.temperature_c
        if self.optimal_temp_min <= temp <= self.optimal_temp_max:
            return 1.0
        deviation = max(self.optimal_temp_min - temp, temp - self.optimal_temp_max, 0.0)
        return float(np.clip(1.0 - deviation * 0.01, 0.6, 1.0))

    def get_usable_energy(self) -> float:
        return max(0.0, self.state.energy_j - self.min_soc * self.capacity)

    def update(self, power_flow_w: float, dt: float) -> float:
        """Positive power_flow charges battery; negative discharges battery."""
        perf = self.get_performance_factor()
        max_charge = self.max_charge_power * perf
        max_discharge = self.max_discharge_power * perf

        if power_flow_w > 0.0:
            actual_power = min(power_flow_w, max_charge)
        else:
            actual_power = max(power_flow_w, -max_discharge)

        if actual_power < 0.0 and self.get_usable_energy() <= 0.0:
            actual_power = 0.0

        self.state.energy_j = float(
            np.clip(self.state.energy_j + actual_power * dt, 0.0, self.capacity)
        )
        self.state.state_of_charge = self.state.energy_j / self.capacity
        heat_generation = abs(actual_power) * 0.05
        self.update_temperature(heat_generation, dt)
        return actual_power

    def reset(self) -> None:
        self.state = BatteryState(
            state_of_charge=self.initial_soc,
            energy_j=self.capacity * self.initial_soc,
            temperature_c=25.0,
        )


class FuelSystem:
    """Fuel tank and mass flow limits."""

    def __init__(
        self, config: dict[str, Any], initial_fuel_kg: float | None = None
    ) -> None:
        self.fuel_capacity = float(config["capacity_kg"])
        self.max_flow_rate = float(config["max_flow_rate"])
        self.fuel_density = float(config["density"])
        self.energy_content_mj_kg = float(config["energy_content_mj_kg"])
        self.fuel_remaining = (
            self.fuel_capacity if initial_fuel_kg is None else float(initial_fuel_kg)
        )

    def consume_fuel(self, fuel_flow_kg_s: float, dt: float) -> float:
        flow = float(np.clip(fuel_flow_kg_s, 0.0, self.max_flow_rate))
        consumed = flow * dt
        self.fuel_remaining = max(0.0, self.fuel_remaining - consumed)
        return consumed

    def get_fuel_mass_effect(self) -> float:
        return self.fuel_remaining

    def get_fuel_percentage(self) -> float:
        return 100.0 * self.fuel_remaining / max(self.fuel_capacity, EPSILON)

    def reset(self, fuel_load: float | None = None) -> None:
        self.fuel_remaining = (
            self.fuel_capacity
            if fuel_load is None
            else float(np.clip(fuel_load, 0.0, self.fuel_capacity))
        )


@dataclass
class OvertakeState:
    is_active: bool = False
    time_used: float = 0.0
    cooldown_timer: float = 0.0


class OvertakeMode:
    """Overtake deployment booster replacing classic DRS concept."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.extra_power = float(config["extra_power"])
        self.activation_gap = float(config["activation_gap_seconds"])
        self.duration_limit = float(config["duration_limit"])
        self.cooldown = float(config["cooldown"])
        self.state = OvertakeState()

    def can_activate(self, gap_to_leader: float | None, battery_soc: float) -> bool:
        return (
            gap_to_leader is not None
            and gap_to_leader < self.activation_gap
            and battery_soc > 0.3
            and self.state.cooldown_timer <= 0.0
        )

    def activate(self) -> None:
        if not self.state.is_active:
            self.state.is_active = True
            self.state.time_used = 0.0

    def deactivate(self) -> None:
        self.state.is_active = False
        self.state.cooldown_timer = self.cooldown

    def update(self, dt: float) -> None:
        if self.state.is_active:
            self.state.time_used += dt
            if self.state.time_used >= self.duration_limit:
                self.deactivate()
        if self.state.cooldown_timer > 0.0:
            self.state.cooldown_timer = max(0.0, self.state.cooldown_timer - dt)

    def get_power_boost(self) -> float:
        return self.extra_power if self.state.is_active else 0.0

    def reset(self) -> None:
        self.state = OvertakeState()


class PowertrainModel:
    """Hybrid powertrain model for F1 2026."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self._load_config(config)
        self.ice = ICEModel(self.config["ice"])
        self.mguk = MGUKModel(self.config["mguk"])
        self.has_mguh = bool(self.config.get("has_mguh", False))
        self.mguh = MGUHModel() if self.has_mguh else None
        self.battery = EnergyStore(self.config["battery"])
        self.transmission = Transmission(self.config["transmission"])
        self.fuel_system = FuelSystem(
            self.config["fuel"], initial_fuel_kg=self.config["fuel"]["capacity_kg"]
        )
        self.overtake_mode = OvertakeMode(self.config["overtake_mode"])

        self.wheel_radius = float(self.config.get("wheel_radius_m", 0.33))
        self.drivetrain_tau = float(self.config.get("drivetrain_tau", 0.03))
        self.last_wheel_torque = 0.0
        self.last_dt = 0.01
        self.last_state: dict[str, Any] = {}

    def _load_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return config
        if DEFAULT_POWERTRAIN_CONFIG_PATH.exists():
            with DEFAULT_POWERTRAIN_CONFIG_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        raise FileNotFoundError(
            f"Powertrain config not found at {DEFAULT_POWERTRAIN_CONFIG_PATH.as_posix()}"
        )

    def _deployment_rate(self, strategy: str, throttle: float, rpm: float) -> float:
        soc = self.battery.soc
        if strategy == "max_power":
            rate = 1.0 if soc > 0.2 else 0.5
        elif strategy == "balanced":
            ice_power = self.ice.state.power_w
            target_mguk = ice_power
            rate = min(target_mguk / max(self.mguk.max_motor_power, EPSILON), 1.0)
            if soc < 0.3:
                rate *= soc / 0.3
        elif strategy == "save_energy":
            rate = 0.2
        else:
            rate = 0.5

        if throttle < 0.15:
            rate *= throttle / 0.15
        if rpm < self.ice.idle_rpm * 1.1:
            rate *= 0.5
        return float(np.clip(rate, 0.0, 1.0))

    @staticmethod
    def _harvest_rate(strategy: str, brake: float) -> float:
        if strategy == "aggressive":
            base = 1.0
        elif strategy == "balanced":
            base = 0.7
        elif strategy == "conservative":
            base = 0.4
        else:
            base = 0.5
        return float(np.clip(base * np.clip(brake, 0.0, 1.0), 0.0, 1.0))

    def calculate_energy_recovery(
        self,
        braking: float,
        rpm: float,
        vehicle_speed: float,
        harvest_strategy: str,
        throttle: float,
    ) -> tuple[float, float]:
        harvest_rate = self._harvest_rate(harvest_strategy, braking)
        mguk_regen_torque, mguk_charge = self.mguk.calculate_generator_power(
            harvest_rate, rpm, braking > 0.1
        )
        mguh_charge = 0.0
        if self.mguh is not None:
            exhaust_energy = self.ice.get_exhaust_energy(rpm, throttle=throttle)
            mguh_charge = self.mguh.calculate_harvest(exhaust_energy, rpm)

        total_charge = mguk_charge + mguh_charge
        if vehicle_speed < 5.0:
            total_charge *= vehicle_speed / 5.0
            mguk_regen_torque *= vehicle_speed / 5.0
        return mguk_regen_torque, total_charge

    def calculate_total_power(
        self,
        throttle: float,
        rpm: float,
        deployment_strategy: str,
        gap_to_leader: float | None,
        dt: float,
    ) -> tuple[float, float, float, float, float, float]:
        ice_power, ice_torque, fuel_flow = self.ice.calculate_ice_power(
            rpm, throttle, dt=dt
        )

        if (
            self.overtake_mode.can_activate(gap_to_leader, self.battery.soc)
            and throttle > 0.95
        ):
            self.overtake_mode.activate()

        deploy_rate = self._deployment_rate(deployment_strategy, throttle, rpm)
        mguk_power, mguk_torque, battery_draw = self.mguk.calculate_motor_power(
            deploy_rate, rpm
        )

        # 50:50 target at high-throttle, high-SOC operation.
        if (
            deployment_strategy == "balanced"
            and throttle > 0.7
            and self.battery.soc > 0.25
        ):
            desired = min(ice_power, self.mguk.max_motor_power)
            mguk_power, mguk_torque, battery_draw = self.mguk.calculate_motor_power(
                desired / max(self.mguk.max_motor_power, EPSILON), rpm
            )

        overtake_boost = self.overtake_mode.get_power_boost()
        if overtake_boost > 0.0:
            extra = min(overtake_boost, self.mguk.max_motor_power - mguk_power)
            if extra > 0.0:
                mguk_power += extra
                omega = rpm * 2.0 * np.pi / 60.0
                mguk_torque += extra / max(omega, EPSILON)
                battery_draw += extra / max(self.mguk.efficiency_motor, EPSILON)

        total_power = ice_power + mguk_power
        total_torque = ice_torque + mguk_torque
        battery_flow = -battery_draw
        return total_power, total_torque, fuel_flow, battery_flow, ice_power, mguk_power

    def calculate_wheel_power(
        self,
        throttle: float,
        brake: float,
        rpm: float,
        gear: int | None,
        vehicle_speed: float,
        deployment_strategy: str = "balanced",
        harvest_strategy: str = "balanced",
        gap_to_leader: float | None = None,
        dt: float = 0.01,
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Main API: wheel torque and energy flows."""
        self.last_dt = float(max(dt, 1e-4))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        if gear is not None and 1 <= gear <= self.transmission.num_gears:
            self.transmission.state.current_gear = int(gear)

        if self.fuel_system.fuel_remaining <= 0.0:
            throttle = 0.0

        total_power, total_torque, fuel_flow, battery_flow, ice_power, mguk_power = (
            self.calculate_total_power(
                throttle=throttle,
                rpm=rpm,
                deployment_strategy=deployment_strategy,
                gap_to_leader=gap_to_leader,
                dt=self.last_dt,
            )
        )

        regen_torque, battery_charge = self.calculate_energy_recovery(
            braking=brake,
            rpm=rpm,
            vehicle_speed=vehicle_speed,
            harvest_strategy=harvest_strategy,
            throttle=throttle,
        )

        net_battery_flow = battery_flow + battery_charge
        actual_battery_flow = self.battery.update(net_battery_flow, self.last_dt)
        actual_fuel_consumed = self.fuel_system.consume_fuel(fuel_flow, self.last_dt)

        crank_torque = total_torque + regen_torque
        wheel_torque_raw = self.transmission.calculate_wheel_torque(crank_torque)

        # Smooth output to avoid discontinuities in controls and RL gradients.
        alpha = np.clip(self.last_dt / max(self.drivetrain_tau, EPSILON), 0.0, 1.0)
        wheel_torque = (
            self.last_wheel_torque + (wheel_torque_raw - self.last_wheel_torque) * alpha
        )
        self.last_wheel_torque = wheel_torque

        if self.battery.soc < 0.12 and deployment_strategy == "max_power":
            LOGGER.warning("Battery SOC critically low during max_power deployment.")
        if self.fuel_system.fuel_remaining < 2.0:
            LOGGER.warning("Fuel critically low.")

        state = {
            "gear": self.transmission.current_gear,
            "in_shift": self.transmission.state.in_shift,
            "engine_rpm": rpm,
            "ice_power_w": ice_power,
            "mguk_power_w": mguk_power,
            "total_power_w": total_power,
            "wheel_torque_nm": wheel_torque,
            "fuel_flow_kg_s": fuel_flow,
            "fuel_used_kg": actual_fuel_consumed,
            "fuel_remaining_kg": self.fuel_system.fuel_remaining,
            "battery_soc": self.battery.soc,
            "battery_temp_c": self.battery.state.temperature_c,
            "battery_flow_w": actual_battery_flow,
            "regen_torque_nm": regen_torque,
            "overtake_active": self.overtake_mode.state.is_active,
            "overtake_cooldown_s": self.overtake_mode.state.cooldown_timer,
            "power_split_ice": ice_power / max(ice_power + mguk_power, EPSILON),
            "power_split_electric": mguk_power / max(ice_power + mguk_power, EPSILON),
        }
        self.last_state = state
        return wheel_torque, fuel_flow, actual_battery_flow, state

    def shift_gear(self, direction: int) -> None:
        if direction > 0:
            self.transmission.shift_up()
        elif direction < 0:
            self.transmission.shift_down()

    def get_optimal_gear(
        self, vehicle_speed: float, target: str = "acceleration"
    ) -> int:
        wheel_speed = vehicle_speed / max(self.wheel_radius, 0.1)
        best_gear = 1
        best_score = -np.inf

        for gear in range(1, self.transmission.num_gears + 1):
            rpm = self.transmission.calculate_engine_rpm(wheel_speed, gear)
            rpm = float(np.clip(rpm, self.ice.idle_rpm, self.ice.max_rpm))
            power, torque, _ = self.ice.calculate_ice_power(
                rpm, throttle=1.0, dt=self.last_dt
            )
            wheel_torque = self.transmission.calculate_wheel_torque(torque, gear)
            if target == "acceleration":
                score = wheel_torque
            elif target == "top_speed":
                score = power - abs(rpm - self.ice.max_rpm * 0.92) * 0.05
            else:
                bsfc = self.ice.get_bsfc(rpm, throttle=0.6)
                score = -bsfc

            if score > best_score:
                best_score = score
                best_gear = gear
        return best_gear

    def update(self, dt: float) -> None:
        self.last_dt = float(max(dt, 1e-4))
        self.transmission.update(self.last_dt)
        self.overtake_mode.update(self.last_dt)

    def reset(self, fuel_load: float = 110.0) -> None:
        self.transmission.reset()
        self.battery.reset()
        self.fuel_system.reset(fuel_load=fuel_load)
        self.overtake_mode.reset()
        self.last_wheel_torque = 0.0
        self.last_state = {}

    def get_state(self) -> dict[str, Any]:
        state = {
            "ice": asdict(self.ice.state),
            "transmission": asdict(self.transmission.state),
            "mguk": asdict(self.mguk.state),
            "battery": asdict(self.battery.state),
            "fuel": {
                "remaining_kg": self.fuel_system.fuel_remaining,
                "remaining_percent": self.fuel_system.get_fuel_percentage(),
            },
            "overtake": asdict(self.overtake_mode.state),
            "last_state": self.last_state,
        }
        return state
