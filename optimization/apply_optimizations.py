#!/usr/bin/env python3
"""Apply and verify Phase 2.6 optimization hooks."""

from __future__ import annotations

from optimization.optimizations import VehicleOptimizations
from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


def apply_all_optimizations() -> None:
    """Apply all available optimizations and print a concise status summary."""
    print("Applying optimizations...")

    vehicle = VehicleDynamicsModel()
    vehicle.reset()

    print("  - Adding cached aero lookups...")
    print("  - Preparing vectorized tire path...")
    _ = VehicleOptimizations.vectorize_tire_calculations(vehicle)

    print("  - Enabling optional JIT kernels...")
    summary = VehicleOptimizations.apply_all(vehicle)

    print("  - Building compact RL state layout...")
    packed = VehicleOptimizations.optimize_memory_layout(vehicle)

    print("Optimizations applied!")
    print(f"Numba enabled: {summary.numba_enabled}")
    print(f"Packed state length: {packed.shape[0]}")
    for note in summary.notes:
        print(f"  * {note}")


if __name__ == "__main__":
    apply_all_optimizations()
