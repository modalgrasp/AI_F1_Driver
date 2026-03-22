#!/usr/bin/env python3
"""Optimization helpers for vehicle dynamics runtime performance."""

from __future__ import annotations

import functools
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel

_numba_mod = (
    importlib.import_module("numba") if importlib.util.find_spec("numba") else None
)
NUMBA_AVAILABLE = _numba_mod is not None

if NUMBA_AVAILABLE:
    jit = _numba_mod.jit  # type: ignore[union-attr]
else:  # pragma: no cover - optional dependency

    def jit(
        *_args: Any, **_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _wrap


@dataclass
class OptimizationSummary:
    cache_enabled: bool
    numba_enabled: bool
    notes: list[str]


@jit(nopython=True, cache=True)
def _numba_tire_kernel(
    slip_angle: float,
    slip_ratio: float,
    normal_load: float,
    b_lat: float,
    c_lat: float,
    d_lat: float,
    e_lat: float,
) -> float:
    """Simple nopython kernel used as a reference hot-loop micro-kernel."""
    alpha = min(max(slip_angle, -0.6), 0.6)
    kappa = min(max(slip_ratio, -1.2), 1.5)
    combined = alpha + 0.15 * kappa
    bx = b_lat * combined
    atan_bx = np.arctan(bx)
    term = bx - e_lat * (bx - atan_bx)
    return d_lat * normal_load * np.sin(c_lat * np.arctan(term))


class VehicleOptimizations:
    """Performance optimizations for vehicle dynamics simulation."""

    @staticmethod
    def vectorize_tire_calculations(vehicle: VehicleDynamicsModel) -> np.ndarray:
        """Compute all tire longitudinal/lateral forces with array operations."""
        normal = vehicle.last_normal_loads.astype(np.float64)
        slip_angles = vehicle.last_slip_angles.astype(np.float64)
        slip_ratios = vehicle.last_slip_ratios.astype(np.float64)
        temps = np.asarray(
            [ts.surface_temp for ts in vehicle.state.tire_states], dtype=np.float64
        )
        wear = np.asarray(
            [ts.wear_percentage for ts in vehicle.state.tire_states], dtype=np.float64
        )

        # Keep explicit calls per tire model but batch pre/post processing in NumPy.
        out = np.zeros((4, 3), dtype=np.float64)
        for i in range(4):
            fx, fy, mz = vehicle.tire_models[i].calculate_forces(
                slip_angle=float(slip_angles[i]),
                slip_ratio=float(slip_ratios[i]),
                normal_load=float(normal[i]),
                temperature=float(temps[i]),
                wear=float(wear[i]),
                wheel_speed=float(vehicle.state.wheel_speeds[i]),
                vehicle_speed=float(vehicle.state.vx),
                camber_angle=-0.05 if i < 2 else -0.04,
            )
            out[i, 0] = float(fx)
            out[i, 1] = float(fy)
            out[i, 2] = float(mz)
        return out

    @staticmethod
    def cache_expensive_calculations(vehicle: VehicleDynamicsModel) -> None:
        """Attach quantized cached wrappers for repeated aero coefficient lookups."""

        aero = vehicle.aero_model

        @functools.lru_cache(maxsize=256)
        def _cached_lift(mode: str) -> tuple[float, float]:
            return aero.get_lift_coefficients(mode)

        @functools.lru_cache(maxsize=256)
        def _cached_drag(mode: str) -> float:
            return aero.get_drag_coefficient(mode)

        aero._cached_lift_coefficients = _cached_lift  # type: ignore[attr-defined]
        aero._cached_drag_coefficient = _cached_drag  # type: ignore[attr-defined]

    @staticmethod
    def use_numba_jit() -> bool:
        """Return whether Numba-backed kernels are available."""
        return NUMBA_AVAILABLE

    @staticmethod
    def optimize_memory_layout(vehicle: VehicleDynamicsModel) -> np.ndarray:
        """Create a compact contiguous state vector suitable for batched RL storage."""
        return np.asarray(
            [
                vehicle.state.x,
                vehicle.state.y,
                vehicle.state.yaw,
                vehicle.state.vx,
                vehicle.state.vy,
                vehicle.state.omega,
                vehicle.state.ax,
                vehicle.state.ay,
                vehicle.state.roll,
                vehicle.state.pitch,
                vehicle.state.engine_rpm,
                float(vehicle.state.gear),
                vehicle.state.battery_soc,
                vehicle.state.fuel_mass,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def batch_computations(
        vehicles: list[VehicleDynamicsModel],
        dt: float,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Step N vehicles and return compact next-state array for vectorized RL loops."""
        if len(vehicles) != int(actions.shape[0]):
            raise ValueError("actions row count must match number of vehicles")

        state_batch = np.zeros((len(vehicles), 14), dtype=np.float64)
        for i, vehicle in enumerate(vehicles):
            steering, throttle, brake, aero_flag = actions[i]
            aero_mode = "high_downforce" if aero_flag >= 0.5 else "low_drag"
            vehicle.update(
                float(dt), float(steering), float(throttle), float(brake), aero_mode
            )
            state_batch[i] = VehicleOptimizations.optimize_memory_layout(vehicle)

        return state_batch

    @staticmethod
    def apply_all(vehicle: VehicleDynamicsModel) -> OptimizationSummary:
        """Apply available optimization hooks to a vehicle model instance."""
        notes: list[str] = []
        VehicleOptimizations.cache_expensive_calculations(vehicle)
        notes.append("Applied aero coefficient cache wrappers")

        if NUMBA_AVAILABLE:
            _ = _numba_tire_kernel(0.05, 0.02, 5000.0, 10.5, 1.35, 1.9, -0.2)
            notes.append("Numba tire kernel compiled")
        else:
            notes.append("Numba not available; using pure-Python fallback")

        return OptimizationSummary(
            cache_enabled=True,
            numba_enabled=NUMBA_AVAILABLE,
            notes=notes,
        )
