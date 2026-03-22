#!/usr/bin/env python3
"""Integrated vehicle dynamics model for F1 2026 simulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vehicle_dynamics.aerodynamics import AerodynamicsModel
from vehicle_dynamics.powertrain import PowertrainModel
from vehicle_dynamics.tire_model import TireModel, TireState

EPSILON = 1e-9
DEFAULT_VEHICLE_CONFIG_PATH = Path("configs/vehicle_config.json")


@dataclass
class VehicleParameters:
    """Physical and numerical parameters for the integrated model."""

    mass: float = 798.0
    fuel_mass: float = 110.0
    wheelbase: float = 3.6
    track_width_front: float = 1.6
    track_width_rear: float = 1.6
    cg_height: float = 0.35
    cg_to_front: float = 1.6
    cg_to_rear: float = 2.0
    iz: float = 1500.0
    ix: float = 300.0
    iy: float = 1800.0
    roll_stiffness: float = 100000.0
    pitch_stiffness: float = 150000.0
    roll_damping: float = 5000.0
    pitch_damping: float = 7000.0
    arb_front: float = 50000.0
    arb_rear: float = 60000.0
    wheel_radius: float = 0.33
    wheel_inertia: float = 1.2
    wheelbase_bias_front: float = 0.55
    integration_method: str = "RK4"
    max_dt: float = 0.001
    tolerance: float = 1e-6
    ride_height_front_nominal: float = 0.050
    ride_height_rear_nominal: float = 0.055


@dataclass
class VehicleState:
    """Vehicle state for 6-DOF planar dynamics plus suspension/tire states."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    wheel_speeds: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    roll: float = 0.0
    pitch: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    tire_states: list[TireState] = field(
        default_factory=lambda: [TireState() for _ in range(4)]
    )
    engine_rpm: float = 5000.0
    gear: int = 1
    battery_soc: float = 0.5
    fuel_mass: float = 110.0


class VehicleDynamicsModel:
    """Complete F1 vehicle dynamics simulation with subsystem integration."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self._load_config(config)
        self.params = self._build_params(self.config)
        self.state = VehicleState(fuel_mass=self.params.fuel_mass)

        self.aero_model = AerodynamicsModel()
        self.powertrain = PowertrainModel()
        # Front tires slightly harder by default for stability, rear softer for traction.
        self.tire_models = [
            TireModel("medium"),
            TireModel("medium"),
            TireModel("soft"),
            TireModel("soft"),
        ]

        self.current_aero_mode = "high_downforce"
        self.ambient_temp = 32.0
        self.track_temp = 45.0

        self.last_normal_loads = np.zeros(4, dtype=np.float64)
        self.last_slip_angles = np.zeros(4, dtype=np.float64)
        self.last_slip_ratios = np.zeros(4, dtype=np.float64)
        self.last_tire_forces = np.zeros((4, 3), dtype=np.float64)
        self.last_aero_forces = {
            "front_downforce": 0.0,
            "rear_downforce": 0.0,
            "drag": 0.0,
            "pitch_moment": 0.0,
            "roll_moment": 0.0,
            "yaw_moment": 0.0,
        }

    def _load_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return config
        if DEFAULT_VEHICLE_CONFIG_PATH.exists():
            with DEFAULT_VEHICLE_CONFIG_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        raise FileNotFoundError(
            f"Vehicle config not found at {DEFAULT_VEHICLE_CONFIG_PATH.as_posix()}"
        )

    @staticmethod
    def _build_params(config: dict[str, Any]) -> VehicleParameters:
        mass_cfg = config.get("mass", {})
        dim_cfg = config.get("dimensions", {})
        inertia_cfg = config.get("inertia", {})
        susp_cfg = config.get("suspension", {})
        integ_cfg = config.get("integration", {})

        cg_to_front = float(dim_cfg.get("cg_to_front", 1.6))
        wheelbase = float(dim_cfg.get("wheelbase", 3.6))
        return VehicleParameters(
            mass=float(mass_cfg.get("car_mass", 798.0)),
            fuel_mass=float(mass_cfg.get("fuel_capacity", 110.0)),
            wheelbase=wheelbase,
            track_width_front=float(dim_cfg.get("track_front", 1.6)),
            track_width_rear=float(dim_cfg.get("track_rear", 1.6)),
            cg_height=float(dim_cfg.get("cg_height", 0.35)),
            cg_to_front=cg_to_front,
            cg_to_rear=max(wheelbase - cg_to_front, 0.1),
            iz=float(inertia_cfg.get("Iz", 1500.0)),
            ix=float(inertia_cfg.get("Ix", 300.0)),
            iy=float(inertia_cfg.get("Iy", 1800.0)),
            roll_stiffness=float(susp_cfg.get("roll_stiffness", 100000.0)),
            pitch_stiffness=float(susp_cfg.get("pitch_stiffness", 150000.0)),
            roll_damping=float(susp_cfg.get("roll_damping", 5000.0)),
            pitch_damping=float(susp_cfg.get("pitch_damping", 7000.0)),
            arb_front=float(susp_cfg.get("arb_front", 50000.0)),
            arb_rear=float(susp_cfg.get("arb_rear", 60000.0)),
            integration_method=str(integ_cfg.get("method", "RK4")),
            max_dt=float(integ_cfg.get("max_dt", 0.001)),
            tolerance=float(integ_cfg.get("tolerance", 1e-6)),
        )

    def get_front_ride_height(self) -> float:
        return float(
            np.clip(
                self.params.ride_height_front_nominal
                - 0.15 * self.state.pitch
                + 0.03 * abs(self.state.roll),
                0.02,
                0.12,
            )
        )

    def get_rear_ride_height(self) -> float:
        return float(
            np.clip(
                self.params.ride_height_rear_nominal
                + 0.15 * self.state.pitch
                + 0.03 * abs(self.state.roll),
                0.02,
                0.12,
            )
        )

    def _mass_total(self, fuel_mass: float | None = None) -> float:
        fm = self.state.fuel_mass if fuel_mass is None else fuel_mass
        return self.params.mass + fm

    def calculate_normal_loads(
        self,
        ax: float,
        ay: float,
        vx: float,
        roll: float,
        pitch: float,
    ) -> np.ndarray:
        """Normal loads on FL, FR, RL, RR including transfer and downforce."""
        total_mass = self._mass_total()
        total_weight = total_mass * 9.81

        weight_front = total_weight * (self.params.cg_to_rear / self.params.wheelbase)
        weight_rear = total_weight * (self.params.cg_to_front / self.params.wheelbase)

        delta_fz_long = total_mass * ax * self.params.cg_height / self.params.wheelbase
        weight_front -= delta_fz_long
        weight_rear += delta_fz_long

        delta_fz_lat_front = (
            total_mass * ay * self.params.cg_height / self.params.track_width_front
        )
        delta_fz_lat_rear = (
            total_mass * ay * self.params.cg_height / self.params.track_width_rear
        )

        front_rh = float(
            np.clip(
                self.params.ride_height_front_nominal - 0.15 * pitch + 0.03 * abs(roll),
                0.02,
                0.12,
            )
        )
        rear_rh = float(
            np.clip(
                self.params.ride_height_rear_nominal + 0.15 * pitch + 0.03 * abs(roll),
                0.02,
                0.12,
            )
        )
        df_front, df_rear = self.aero_model.calculate_downforce(
            speed=max(vx, 0.0),
            aero_mode=self.current_aero_mode,
            ride_height_front=front_rh,
            ride_height_rear=rear_rh,
            pitch=pitch,
            roll=roll,
        )

        # Roll distribution from suspension and anti-roll bars.
        roll_front = (self.params.roll_stiffness * 0.45 + self.params.arb_front) * roll
        roll_rear = (self.params.roll_stiffness * 0.55 + self.params.arb_rear) * roll

        fz_fl = (
            weight_front * 0.5
            + df_front * 0.5
            - delta_fz_lat_front
            - roll_front / max(self.params.track_width_front, 0.1)
        )
        fz_fr = (
            weight_front * 0.5
            + df_front * 0.5
            + delta_fz_lat_front
            + roll_front / max(self.params.track_width_front, 0.1)
        )
        fz_rl = (
            weight_rear * 0.5
            + df_rear * 0.5
            - delta_fz_lat_rear
            - roll_rear / max(self.params.track_width_rear, 0.1)
        )
        fz_rr = (
            weight_rear * 0.5
            + df_rear * 0.5
            + delta_fz_lat_rear
            + roll_rear / max(self.params.track_width_rear, 0.1)
        )

        loads = np.clip(
            np.asarray([fz_fl, fz_fr, fz_rl, fz_rr], dtype=np.float64), 20.0, None
        )
        return loads

    def calculate_slip_angles(
        self, vx: float, vy: float, omega: float, steering_angle: float
    ) -> np.ndarray:
        """Slip angle for each tire contact patch."""
        vx_safe = np.sign(vx) * max(abs(vx), 0.5)

        vy_fl = vy + omega * self.params.cg_to_front
        vy_fr = vy + omega * self.params.cg_to_front
        vy_rl = vy - omega * self.params.cg_to_rear
        vy_rr = vy - omega * self.params.cg_to_rear

        alpha_fl = np.arctan2(vy_fl, vx_safe) - steering_angle
        alpha_fr = np.arctan2(vy_fr, vx_safe) - steering_angle
        alpha_rl = np.arctan2(vy_rl, vx_safe)
        alpha_rr = np.arctan2(vy_rr, vx_safe)
        return np.asarray([alpha_fl, alpha_fr, alpha_rl, alpha_rr], dtype=np.float64)

    def calculate_slip_ratios(self, vx: float, wheel_speeds: np.ndarray) -> np.ndarray:
        """Longitudinal slip ratios for each wheel."""
        wheel_linear = wheel_speeds * self.params.wheel_radius
        denom = max(abs(vx), 1.0)
        slip = (wheel_linear - vx) / denom
        return np.clip(slip, -1.2, 2.5)

    @staticmethod
    def _transform_tire_force(
        fx: float, fy: float, steering_angle: float
    ) -> tuple[float, float]:
        c_val = np.cos(steering_angle)
        s_val = np.sin(steering_angle)
        fx_body = fx * c_val - fy * s_val
        fy_body = fx * s_val + fy * c_val
        return float(fx_body), float(fy_body)

    def calculate_tire_forces(
        self,
        vx: float,
        vy: float,
        omega: float,
        ax: float,
        ay: float,
        roll: float,
        pitch: float,
        steering_angle: float,
        wheel_speeds: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate all tire forces and yaw moment."""
        normal_loads = self.calculate_normal_loads(ax, ay, vx, roll, pitch)
        slip_angles = self.calculate_slip_angles(vx, vy, omega, steering_angle)
        slip_ratios = self.calculate_slip_ratios(vx, wheel_speeds)

        per_tire = np.zeros((4, 3), dtype=np.float64)
        for i in range(4):
            tire_state = self.state.tire_states[i]
            alpha_i = float(
                np.clip(slip_angles[i], -np.deg2rad(20.0), np.deg2rad(20.0))
            )
            kappa_i = float(np.clip(slip_ratios[i], -1.5, 1.5))
            fx_i, fy_i, mz_i = self.tire_models[i].calculate_forces(
                slip_angle=alpha_i,
                slip_ratio=kappa_i,
                normal_load=normal_loads[i],
                temperature=tire_state.surface_temp,
                wear=tire_state.wear_percentage,
                wheel_speed=wheel_speeds[i],
                vehicle_speed=vx,
                camber_angle=-0.05 if i < 2 else -0.04,
            )
            per_tire[i, 0] = float(fx_i)
            per_tire[i, 1] = float(fy_i)
            per_tire[i, 2] = float(mz_i)

        fx_fl, fy_fl = self._transform_tire_force(
            per_tire[0, 0], per_tire[0, 1], steering_angle
        )
        fx_fr, fy_fr = self._transform_tire_force(
            per_tire[1, 0], per_tire[1, 1], steering_angle
        )
        fx_rl, fy_rl = float(per_tire[2, 0]), float(per_tire[2, 1])
        fx_rr, fy_rr = float(per_tire[3, 0]), float(per_tire[3, 1])

        fx_total = fx_fl + fx_fr + fx_rl + fx_rr
        fy_total = fy_fl + fy_fr + fy_rl + fy_rr

        twf = self.params.track_width_front * 0.5
        twr = self.params.track_width_rear * 0.5
        mz_total = (
            fx_fl * (-twf)
            + fy_fl * self.params.cg_to_front
            + fx_fr * (twf)
            + fy_fr * self.params.cg_to_front
            + fx_rl * (-twr)
            - fy_rl * self.params.cg_to_rear
            + fx_rr * (twr)
            - fy_rr * self.params.cg_to_rear
            + np.sum(per_tire[:, 2])
        )

        return (
            float(fx_total),
            float(fy_total),
            float(mz_total),
            normal_loads,
            slip_angles,
            slip_ratios,
            per_tire,
        )

    def _powertrain_step(
        self, dt: float, throttle: float, brake: float, speed: float
    ) -> dict[str, Any]:
        """Advance powertrain once per integrator step to avoid RK side effects."""
        target_gear = self.powertrain.get_optimal_gear(speed, target="acceleration")
        if (
            target_gear > self.powertrain.transmission.current_gear
            and self.state.engine_rpm > 13000.0
        ):
            self.powertrain.shift_gear(+1)
        elif (
            target_gear < self.powertrain.transmission.current_gear
            and self.state.engine_rpm < 7000.0
        ):
            self.powertrain.shift_gear(-1)

        self.powertrain.update(dt)
        wheel_torque, fuel_flow, battery_flow, pt_state = (
            self.powertrain.calculate_wheel_power(
                throttle=throttle,
                brake=brake,
                rpm=self.state.engine_rpm,
                gear=self.powertrain.transmission.current_gear,
                vehicle_speed=speed,
                deployment_strategy="balanced",
                harvest_strategy="balanced",
                dt=dt,
            )
        )
        return {
            "wheel_torque": float(wheel_torque),
            "fuel_flow": float(fuel_flow),
            "battery_flow": float(battery_flow),
            "state": pt_state,
        }

    def _derivatives(
        self,
        vec: np.ndarray,
        controls: tuple[float, float, float],
    ) -> np.ndarray:
        x, y, yaw, vx, vy, omega = vec
        steering, throttle, brake = controls
        _ = throttle
        _ = brake

        (
            fx_tires,
            fy_tires,
            mz_tires,
            normal_loads,
            slip_angles,
            slip_ratios,
            per_tire,
        ) = self.calculate_tire_forces(
            vx=vx,
            vy=vy,
            omega=omega,
            ax=self.state.ax,
            ay=self.state.ay,
            roll=self.state.roll,
            pitch=self.state.pitch,
            steering_angle=steering,
            wheel_speeds=self.state.wheel_speeds,
        )

        df_front, df_rear, drag, pitch_moment, roll_moment, yaw_moment_aero = (
            self.aero_model.calculate_forces(
                speed=max(vx, 0.0),
                aero_mode=self.current_aero_mode,
                ride_height_front=self.get_front_ride_height(),
                ride_height_rear=self.get_rear_ride_height(),
                pitch=self.state.pitch,
                roll=self.state.roll,
                yaw=yaw,
            )
        )
        self.last_aero_forces = {
            "front_downforce": df_front,
            "rear_downforce": df_rear,
            "drag": drag,
            "pitch_moment": pitch_moment,
            "roll_moment": roll_moment,
            "yaw_moment": yaw_moment_aero,
        }

        fx_total = fx_tires - drag
        fy_total = fy_tires
        mz_total = mz_tires + yaw_moment_aero

        m = self._mass_total()
        ax = fx_total / m + vy * omega
        ay = fy_total / m - vx * omega
        omega_dot = mz_total / max(self.params.iz, 1.0)

        vx_global = vx * np.cos(yaw) - vy * np.sin(yaw)
        vy_global = vx * np.sin(yaw) + vy * np.cos(yaw)

        self.last_normal_loads = normal_loads
        self.last_slip_angles = slip_angles
        self.last_slip_ratios = slip_ratios
        self.last_tire_forces = per_tire

        return np.asarray(
            [vx_global, vy_global, omega, ax, ay, omega_dot], dtype=np.float64
        )

    def integrate_state(
        self, dt: float, steering: float, throttle: float, brake: float
    ) -> None:
        """RK4 integration for x, y, yaw, vx, vy, omega."""
        state_vec = np.asarray(
            [
                self.state.x,
                self.state.y,
                self.state.yaw,
                self.state.vx,
                self.state.vy,
                self.state.omega,
            ],
            dtype=np.float64,
        )
        controls = (steering, throttle, brake)

        k1 = self._derivatives(state_vec, controls)
        k2 = self._derivatives(state_vec + 0.5 * dt * k1, controls)
        k3 = self._derivatives(state_vec + 0.5 * dt * k2, controls)
        k4 = self._derivatives(state_vec + dt * k3, controls)
        new_state = state_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.state.x = float(new_state[0])
        self.state.y = float(new_state[1])
        self.state.yaw = float(np.arctan2(np.sin(new_state[2]), np.cos(new_state[2])))
        self.state.vx = float(new_state[3])
        self.state.vy = float(new_state[4])
        self.state.omega = float(new_state[5])
        self.state.ax = float((k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0)
        self.state.ay = float((k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0)

    def update_suspension(self, dt: float) -> None:
        """Update roll and pitch states from acceleration and aero moments."""
        m = self._mass_total()
        roll_m = (
            m * self.state.ay * self.params.cg_height
            + self.last_aero_forces["roll_moment"]
        )
        pitch_m = (
            -m * self.state.ax * self.params.cg_height
            + self.last_aero_forces["pitch_moment"]
        )

        roll_acc = (
            roll_m
            - self.params.roll_stiffness * self.state.roll
            - self.params.roll_damping * self.state.roll_rate
        ) / max(self.params.ix, 1.0)
        pitch_acc = (
            pitch_m
            - self.params.pitch_stiffness * self.state.pitch
            - self.params.pitch_damping * self.state.pitch_rate
        ) / max(self.params.iy, 1.0)

        self.state.roll_rate += roll_acc * dt
        self.state.pitch_rate += pitch_acc * dt
        self.state.roll += self.state.roll_rate * dt
        self.state.pitch += self.state.pitch_rate * dt

        self.state.roll = float(np.clip(self.state.roll, -0.25, 0.25))
        self.state.pitch = float(np.clip(self.state.pitch, -0.2, 0.2))

    def update_wheels(
        self, dt: float, throttle: float, brake: float, drive_torque_total: float
    ) -> None:
        """Update wheel angular speeds with drive/brake and tire reaction torques."""
        _ = throttle
        wheel_inertia = self.params.wheel_inertia
        tire_reaction = self.last_tire_forces[:, 0] * self.params.wheel_radius
        brake_torque = 3500.0 * np.clip(brake, 0.0, 1.0)

        drive = np.zeros(4, dtype=np.float64)
        # Rear-wheel drive split.
        drive[2] = 0.5 * drive_torque_total
        drive[3] = 0.5 * drive_torque_total

        brake_split_front = 0.58
        braking = np.asarray(
            [
                brake_torque * brake_split_front * 0.5,
                brake_torque * brake_split_front * 0.5,
                brake_torque * (1.0 - brake_split_front) * 0.5,
                brake_torque * (1.0 - brake_split_front) * 0.5,
            ],
            dtype=np.float64,
        )

        wheel_acc = (drive - braking - tire_reaction) / max(wheel_inertia, 0.1)
        self.state.wheel_speeds += wheel_acc * dt

        rolling_speed = max(self.state.vx, 0.0) / max(self.params.wheel_radius, 0.1)
        # Keep wheel speeds close to rolling kinematics to avoid artificial drag at launch.
        self.state.wheel_speeds[0:2] += (
            rolling_speed - self.state.wheel_speeds[0:2]
        ) * np.clip(8.0 * dt, 0.0, 1.0)
        rear_relax = np.clip((6.0 + 6.0 * throttle) * dt, 0.0, 1.0)
        self.state.wheel_speeds[2:4] += (
            rolling_speed - self.state.wheel_speeds[2:4]
        ) * rear_relax
        if throttle > 0.3 and brake < 0.1:
            self.state.wheel_speeds[2:4] = np.maximum(
                self.state.wheel_speeds[2:4], 1.02 * rolling_speed
            )

        self.state.wheel_speeds = np.clip(self.state.wheel_speeds, 0.0, 1200.0)

    def update_tires(self, dt: float) -> None:
        """Advance tire thermal/wear states using latest force/slip estimates."""
        for i, tire in enumerate(self.tire_models):
            fx = float(self.last_tire_forces[i, 0])
            fy = float(self.last_tire_forces[i, 1])
            alpha = float(self.last_slip_angles[i])
            kappa = float(self.last_slip_ratios[i])
            vx_slip = float(
                self.state.wheel_speeds[i] * self.params.wheel_radius - self.state.vx
            )
            vy_slip = float(
                self.state.vy
                + (self.params.cg_to_front if i < 2 else -self.params.cg_to_rear)
                * self.state.omega
            )

            tire.update_temperature(
                forces={"Fx": fx, "Fy": fy},
                speeds={
                    "vx_slip": vx_slip,
                    "vy_slip": vy_slip,
                    "track_temp": self.track_temp,
                    "camber": -0.05 if i < 2 else -0.04,
                },
                ambient_temp=self.ambient_temp,
                dt=dt,
            )
            tire.update_wear(
                slip_angle=alpha,
                slip_ratio=kappa,
                temperature=tire.state.surface_temp,
                dt=dt,
            )
            self.state.tire_states[i] = tire.state

    def update(
        self,
        dt: float,
        steering: float,
        throttle: float,
        brake: float,
        aero_mode: str = "high_downforce",
    ) -> VehicleState:
        """Advance simulation by dt and return updated state."""
        dt = float(np.clip(dt, 1e-5, 0.1))
        steering = float(np.clip(steering, -0.8, 0.8))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        self.current_aero_mode = aero_mode

        # Numerical stability via substepping.
        n_steps = int(np.ceil(dt / max(self.params.max_dt, 1e-4)))
        sub_dt = dt / n_steps

        for _ in range(n_steps):
            self.aero_model.update(sub_dt)

            speed = float(np.hypot(self.state.vx, self.state.vy))
            self.state.engine_rpm = float(
                np.clip(
                    self.powertrain.transmission.calculate_engine_rpm(
                        speed / max(self.params.wheel_radius, 0.1),
                        self.powertrain.transmission.current_gear,
                    ),
                    self.powertrain.ice.idle_rpm,
                    self.powertrain.ice.max_rpm,
                )
            )

            pt = self._powertrain_step(sub_dt, throttle, brake, speed)

            self.integrate_state(sub_dt, steering, throttle, brake)

            self.update_wheels(sub_dt, throttle, brake, pt["wheel_torque"])
            self.update_tires(sub_dt)
            self.update_suspension(sub_dt)

            self.state.gear = int(pt["state"]["gear"])
            self.state.battery_soc = float(pt["state"]["battery_soc"])
            self.state.fuel_mass = float(pt["state"]["fuel_remaining_kg"])

        return self.state

    def get_speed_kmh(self) -> float:
        speed_ms = float(np.hypot(self.state.vx, self.state.vy))
        return speed_ms * 3.6

    def get_slip_angle_deg(self) -> float:
        if abs(self.state.vx) > 0.1:
            return float(np.degrees(np.arctan2(self.state.vy, self.state.vx)))
        return 0.0

    def get_g_forces(self) -> tuple[float, float]:
        g = 9.81
        return self.state.ax / g, self.state.ay / g

    def get_state(self) -> dict[str, Any]:
        return {
            "x": self.state.x,
            "y": self.state.y,
            "yaw": self.state.yaw,
            "vx": self.state.vx,
            "vy": self.state.vy,
            "omega": self.state.omega,
            "ax": self.state.ax,
            "ay": self.state.ay,
            "roll": self.state.roll,
            "pitch": self.state.pitch,
            "engine_rpm": self.state.engine_rpm,
            "gear": self.state.gear,
            "battery_soc": self.state.battery_soc,
            "fuel_mass": self.state.fuel_mass,
            "speed_kmh": self.get_speed_kmh(),
            "slip_angle_deg": self.get_slip_angle_deg(),
            "g_forces": self.get_g_forces(),
            "normal_loads": self.last_normal_loads.tolist(),
            "slip_angles": self.last_slip_angles.tolist(),
            "slip_ratios": self.last_slip_ratios.tolist(),
            "tire_forces": self.last_tire_forces.tolist(),
            "aero_forces": self.last_aero_forces,
            "powertrain": self.powertrain.get_state(),
            "tire_states": [asdict(ts) for ts in self.state.tire_states],
        }

    def reset(self, fuel_load: float = 110.0) -> VehicleState:
        self.state = VehicleState(
            fuel_mass=float(np.clip(fuel_load, 0.0, self.params.fuel_mass))
        )
        self.powertrain.reset(fuel_load=fuel_load)
        self.aero_model.reset()
        self.current_aero_mode = "high_downforce"

        compounds = ["medium", "medium", "soft", "soft"]
        for tire, compound in zip(self.tire_models, compounds):
            tire.reset(compound)
        self.state.tire_states = [tm.state for tm in self.tire_models]

        self.last_normal_loads = np.zeros(4, dtype=np.float64)
        self.last_slip_angles = np.zeros(4, dtype=np.float64)
        self.last_slip_ratios = np.zeros(4, dtype=np.float64)
        self.last_tire_forces = np.zeros((4, 3), dtype=np.float64)
        self.last_aero_forces = {
            "front_downforce": 0.0,
            "rear_downforce": 0.0,
            "drag": 0.0,
            "pitch_moment": 0.0,
            "roll_moment": 0.0,
            "yaw_moment": 0.0,
        }
        return self.state
