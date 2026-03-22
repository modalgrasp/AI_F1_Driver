#!/usr/bin/env python3
"""Performance and memory regression tests for CI."""

from __future__ import annotations

import os
import time
import tracemalloc

import pytest

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


@pytest.mark.slow
@pytest.mark.integration
def test_performance_regression() -> None:
    vehicle = VehicleDynamicsModel()
    vehicle.reset()

    start = time.perf_counter()
    num_steps = 5000
    for _ in range(num_steps):
        vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")

    elapsed = time.perf_counter() - start
    steps_per_sec = num_steps / max(elapsed, 1e-12)

    # Conservative default for broad CI hardware; override in dedicated perf
    # runners to enforce stricter targets (for example 1000+).
    min_perf = float(os.getenv("F1_MIN_STEPS_PER_SEC", "150"))
    assert (
        steps_per_sec >= min_perf
    ), f"Performance regression: {steps_per_sec:.1f} steps/sec < {min_perf:.1f}"


@pytest.mark.slow
@pytest.mark.integration
def test_parallel_env_throughput() -> None:
    num_envs = 8
    vehicles = [VehicleDynamicsModel() for _ in range(num_envs)]
    for vehicle in vehicles:
        vehicle.reset()

    num_steps = 1500
    start = time.perf_counter()
    for _ in range(num_steps):
        for vehicle in vehicles:
            vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")
    elapsed = time.perf_counter() - start

    total_steps = num_steps * num_envs
    steps_per_sec = total_steps / max(elapsed, 1e-12)

    # Parallel throughput varies widely by host CPU; keep defaults stable and
    # enforce production targets via environment configuration.
    min_total_perf = float(os.getenv("F1_MIN_PARALLEL_STEPS_PER_SEC", "180"))
    assert (
        steps_per_sec >= min_total_perf
    ), f"Parallel throughput regression: {steps_per_sec:.1f} < {min_total_perf:.1f} steps/sec"


@pytest.mark.slow
@pytest.mark.integration
def test_memory_leak() -> None:
    tracemalloc.start()

    vehicle = VehicleDynamicsModel()
    vehicle.reset()

    for _ in range(5000):
        vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")
    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(5000):
        vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")
    snapshot2 = tracemalloc.take_snapshot()

    stats = snapshot2.compare_to(snapshot1, "lineno")
    total_growth = sum(stat.size_diff for stat in stats)
    tracemalloc.stop()

    max_growth = int(
        float(os.getenv("F1_MAX_MEMORY_GROWTH_BYTES", str(3 * 1024 * 1024)))
    )
    assert (
        total_growth < max_growth
    ), f"Potential memory leak: growth={total_growth / 1024.0:.1f}KB > {max_growth/1024.0:.1f}KB"
