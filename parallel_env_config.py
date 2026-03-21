#!/usr/bin/env python3
"""Parallel environment planning and benchmarking for mixed CPU/GPU workloads."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch


@dataclass
class ParallelRecommendation:
    cpu_workers: int
    gpu_render_batch: int
    env_count: int
    notes: list[str]


def recommend() -> ParallelRecommendation:
    logical = psutil.cpu_count(logical=True) or 8
    physical = psutil.cpu_count(logical=False) or max(4, logical // 2)

    gpu_factor = 8
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_factor = 12 if vram_gb >= 12 else 8

    cpu_workers = max(2, min(physical, 10))
    env_count = max(4, min(logical, cpu_workers * 2))
    render_batch = max(2, min(gpu_factor, env_count))

    notes = [
        "Pin simulation processes to physical cores for stable frame pacing.",
        "Use batched policy inference on GPU to reduce launch overhead.",
        "Avoid oversubscribing workers beyond thermal budget.",
    ]
    return ParallelRecommendation(cpu_workers, render_batch, env_count, notes)


def synthetic_benchmark(env_count: int, steps: int = 1000) -> dict:
    # Simulated throughput benchmark for planning before full simulator multiprocess setup.
    t0 = time.perf_counter()
    obs = torch.randn((env_count, 256), device="cuda" if torch.cuda.is_available() else "cpu")
    w = torch.randn((256, 256), device=obs.device)
    for _ in range(steps):
        obs = torch.relu(obs @ w)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "env_count": env_count,
        "steps": steps,
        "seconds": dt,
        "env_steps_per_sec": (env_count * steps) / max(dt, 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel environment configuration recommender")
    parser.add_argument("--output", type=Path, default=Path("logs/parallel_env_recommendation.json"))
    args = parser.parse_args()

    rec = recommend()
    bench = synthetic_benchmark(rec.env_count)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "recommendation": rec.__dict__,
        "benchmark": bench,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
