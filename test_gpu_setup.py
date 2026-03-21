#!/usr/bin/env python3
"""Comprehensive GPU setup validation and benchmark suite."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from gpu_config_manager import GPUConfigManager
from mixed_precision_config import MixedPrecisionManager


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


def test_cuda_available() -> TestResult:
    ok = torch.cuda.is_available()
    return TestResult("cuda_available", ok, f"torch.cuda.is_available={ok}")


def test_gpu_identification() -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("gpu_identification", False, "CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    ok = ("5070" in name or "RTX" in name) and vram >= 10.0
    return TestResult("gpu_identification", ok, f"name={name}, vram_gb={vram:.2f}")


def test_tensor_ops() -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("tensor_ops", False, "CUDA unavailable")
    a = torch.randn((1024, 1024), device="cuda")
    b = torch.randn((1024, 1024), device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    ok = c.shape == (1024, 1024) and torch.isfinite(c).all().item()
    return TestResult("tensor_ops", bool(ok), "matrix multiplication validated")


def test_memory_allocation() -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("memory_allocation", False, "CUDA unavailable")
    try:
        t1 = torch.empty((2048, 2048, 4), device="cuda")
        t2 = torch.empty((2048, 2048, 4), device="cuda")
        del t1, t2
        torch.cuda.empty_cache()
        return TestResult("memory_allocation", True, "allocation and release successful")
    except RuntimeError as exc:
        return TestResult("memory_allocation", False, str(exc))


def test_mixed_precision() -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("mixed_precision", False, "CUDA unavailable")
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 64))
    amp = MixedPrecisionManager(enabled=True, dtype="float16")
    metrics = amp.benchmark_wrapper(model, (256, 256), steps=20)
    ok = metrics.iter_per_sec > 0
    return TestResult("mixed_precision", ok, f"iter_per_sec={metrics.iter_per_sec:.2f}, peak_gb={metrics.peak_memory_gb:.2f}")


def test_multi_env_simulation() -> TestResult:
    # Simulated multi-env loop to verify stability under repeated GPU updates.
    if not torch.cuda.is_available():
        return TestResult("multi_env_simulation", False, "CUDA unavailable")
    envs = 8
    obs = torch.randn((envs, 512), device="cuda")
    w = torch.randn((512, 512), device="cuda")
    for _ in range(100):
        obs = torch.tanh(obs @ w)
    torch.cuda.synchronize()
    return TestResult("multi_env_simulation", True, "simulated parallel env workload passed")


def test_perf_benchmark() -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("performance", False, "CUDA unavailable")
    n = 4096
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(8):
        _ = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops = 2 * (n**3) * 8
    tflops = flops / max(dt, 1e-9) / 1e12
    return TestResult("performance", tflops > 5.0, f"estimated_tflops={tflops:.2f}")


def test_thermal_stability(duration_sec: int) -> TestResult:
    if not torch.cuda.is_available():
        return TestResult("thermal_stability", False, "CUDA unavailable")
    n = 2048
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    end = time.time() + duration_sec
    while time.time() < end:
        _ = a @ b
    torch.cuda.synchronize()
    return TestResult("thermal_stability", True, f"stress run completed for {duration_sec}s")


def run_suite(quick: bool) -> dict:
    tests = [
        test_cuda_available,
        test_gpu_identification,
        test_tensor_ops,
        test_memory_allocation,
        test_mixed_precision,
        test_multi_env_simulation,
        test_perf_benchmark,
    ]
    if quick:
        thermal = lambda: test_thermal_stability(30)
    else:
        thermal = lambda: test_thermal_stability(600)
    tests.append(thermal)

    results = [fn() for fn in tests]
    manager = GPUConfigManager()
    inventory_path = manager.export_inventory()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "quick_mode": quick,
        "results": [asdict(r) for r in results],
        "all_passed": all(r.passed for r in results),
        "inventory": str(inventory_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU setup validation suite")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/gpu_test_report.json"))
    args = parser.parse_args()

    report = run_suite(quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
