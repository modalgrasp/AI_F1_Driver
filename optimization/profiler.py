#!/usr/bin/env python3
"""Profiling utilities for complete vehicle dynamics system."""

from __future__ import annotations

import cProfile
import json
import pstats
import time
import tracemalloc
from io import StringIO
from pathlib import Path
from typing import Any

from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


class PerformanceProfiler:
    """Profile vehicle dynamics performance and memory behavior."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def profile_single_step(self, num_iterations: int = 10000) -> dict[str, Any]:
        vehicle = VehicleDynamicsModel()
        vehicle.reset()

        profiler = cProfile.Profile()
        profiler.enable()

        for _ in range(num_iterations):
            vehicle.update(
                dt=0.001,
                steering=0.1,
                throttle=0.8,
                brake=0.0,
                aero_mode="high_downforce",
            )

        profiler.disable()

        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(50)

        total_time = stats.total_tt
        time_per_step = total_time / max(num_iterations, 1)
        steps_per_second = 1.0 / max(time_per_step, 1e-12)

        self.results["single_step"] = {
            "iterations": int(num_iterations),
            "total_time": float(total_time),
            "time_per_step": float(time_per_step),
            "steps_per_second": float(steps_per_second),
            "profile_output": stream.getvalue(),
        }
        return self.results["single_step"]

    def profile_subsystems(self, num_calls: int = 10000) -> dict[str, Any]:
        vehicle = VehicleDynamicsModel()
        vehicle.reset()
        vehicle.state.vx = 50.0

        start = time.perf_counter()
        for _ in range(num_calls):
            vehicle.tire_models[0].calculate_forces(
                slip_angle=0.1,
                slip_ratio=0.05,
                normal_load=5000.0,
                temperature=100.0,
                wear=20.0,
                wheel_speed=50.0,
                vehicle_speed=50.0,
            )
        tire_time = (time.perf_counter() - start) / max(num_calls, 1)

        start = time.perf_counter()
        for _ in range(num_calls):
            vehicle.aero_model.calculate_downforce(
                speed=50.0,
                aero_mode="high_downforce",
                ride_height_front=0.05,
                ride_height_rear=0.05,
            )
        aero_time = (time.perf_counter() - start) / max(num_calls, 1)

        start = time.perf_counter()
        for _ in range(num_calls):
            vehicle.powertrain.calculate_wheel_power(
                throttle=0.8,
                brake=0.0,
                rpm=11000.0,
                gear=6,
                vehicle_speed=50.0,
                deployment_strategy="balanced",
                harvest_strategy="balanced",
                dt=0.001,
            )
        powertrain_time = (time.perf_counter() - start) / max(num_calls, 1)

        start = time.perf_counter()
        for _ in range(num_calls):
            vehicle.integrate_state(
                dt=0.001,
                steering=0.1,
                throttle=0.8,
                brake=0.0,
            )
        integration_time = (time.perf_counter() - start) / max(num_calls, 1)

        self.results["subsystems"] = {
            "tire_model_us": float(tire_time * 1e6),
            "aero_model_us": float(aero_time * 1e6),
            "powertrain_us": float(powertrain_time * 1e6),
            "integration_us": float(integration_time * 1e6),
        }
        return self.results["subsystems"]

    def profile_memory_usage(self, steps: int = 10000) -> dict[str, Any]:
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        vehicle = VehicleDynamicsModel()
        vehicle.reset()
        snapshot2 = tracemalloc.take_snapshot()

        for _ in range(steps):
            vehicle.update(0.001, 0.1, 0.8, 0.0, "high_downforce")
        snapshot3 = tracemalloc.take_snapshot()

        creation_stats = snapshot2.compare_to(snapshot1, "lineno")
        runtime_stats = snapshot3.compare_to(snapshot2, "lineno")
        total_memory = sum(stat.size for stat in snapshot3.statistics("lineno"))

        self.results["memory"] = {
            "total_bytes": int(total_memory),
            "total_mb": float(total_memory / (1024 * 1024)),
            "creation_top": [str(stat) for stat in creation_stats[:10]],
            "runtime_top": [str(stat) for stat in runtime_stats[:10]],
        }

        tracemalloc.stop()
        return self.results["memory"]

    def generate_report(
        self, output_path: str = "optimization/profile_report.md"
    ) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = ["# Performance Profiling Report\n\n"]
        if "single_step" in self.results:
            single = self.results["single_step"]
            lines.append("## Single Step Performance\n")
            lines.append(f"- Iterations: {single['iterations']}\n")
            lines.append(f"- Steps per second: {single['steps_per_second']:.1f}\n")
            lines.append(f"- Time per step: {single['time_per_step']*1000:.4f} ms\n\n")

        if "subsystems" in self.results:
            lines.append("## Subsystem Breakdown\n")
            for name, value in self.results["subsystems"].items():
                lines.append(f"- {name}: {value:.2f} us\n")
            lines.append("\n")

        if "memory" in self.results:
            memory = self.results["memory"]
            lines.append("## Memory Usage\n")
            lines.append(f"- Total: {memory['total_mb']:.3f} MB\n\n")

        out_path.write_text("".join(lines), encoding="utf-8")
        return out_path

    def save_results_json(
        self, output_path: str = "optimization/profile_results.json"
    ) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        return out_path


if __name__ == "__main__":
    profiler = PerformanceProfiler()
    profiler.profile_single_step(5000)
    profiler.profile_subsystems(5000)
    profiler.profile_memory_usage(5000)
    report = profiler.generate_report()
    payload = profiler.save_results_json()
    print(f"Profiling report written to: {report}")
    print(f"Profiling json written to: {payload}")
