#!/usr/bin/env python3
"""Benchmark suite for vehicle dynamics performance validation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


class PerformanceBenchmarks:
    """Standard benchmarks for comparing runtime characteristics."""

    def benchmark_single_step(
        self, vehicle: VehicleDynamicsModel, num_iterations: int = 100000
    ) -> dict[str, Any]:
        start = time.perf_counter()
        for _ in range(num_iterations):
            vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")

        elapsed = time.perf_counter() - start
        steps_per_sec = num_iterations / max(elapsed, 1e-12)

        return {
            "iterations": int(num_iterations),
            "elapsed_seconds": float(elapsed),
            "steps_per_second": float(steps_per_sec),
            "ms_per_step": float((elapsed / max(num_iterations, 1)) * 1000.0),
        }

    def benchmark_parallel_environments(
        self, num_envs: int = 8, num_steps: int = 10000
    ) -> dict[str, Any]:
        vehicles = [VehicleDynamicsModel() for _ in range(num_envs)]
        for vehicle in vehicles:
            vehicle.reset()

        start = time.perf_counter()
        for _ in range(num_steps):
            for vehicle in vehicles:
                vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")

        elapsed = time.perf_counter() - start
        total_steps = num_steps * num_envs
        steps_per_sec = total_steps / max(elapsed, 1e-12)

        return {
            "num_environments": int(num_envs),
            "steps_per_environment": int(num_steps),
            "total_steps": int(total_steps),
            "elapsed_seconds": float(elapsed),
            "steps_per_second": float(steps_per_sec),
        }

    def compare_before_after(self) -> dict[str, dict[str, Any]]:
        return {"before": {}, "after": {}}

    def run_and_save(
        self, output_path: str = "optimization/benchmark_results.json"
    ) -> dict[str, Any]:
        vehicle = VehicleDynamicsModel()
        vehicle.reset()

        single = self.benchmark_single_step(vehicle, num_iterations=10000)
        parallel = self.benchmark_parallel_environments(num_envs=8, num_steps=2000)

        payload = {
            "single_step": single,
            "parallel": parallel,
            "steps_per_second": single["steps_per_second"],
            "ms_per_step": single["ms_per_step"],
            "memory_mb": None,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


if __name__ == "__main__":
    bench = PerformanceBenchmarks()
    result = bench.run_and_save()
    print(json.dumps(result, indent=2))
