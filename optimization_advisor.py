#!/usr/bin/env python3
"""Analyze GPU reports and generate actionable optimization recommendations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def advise(batch_report: dict[str, Any], gpu_test_report: dict[str, Any], parallel_report: dict[str, Any]) -> dict[str, Any]:
    recommendations: list[str] = []
    snippets: list[str] = []

    batch_size = batch_report.get("optimal_batch_size", 0)
    sps = batch_report.get("samples_per_sec", 0.0)
    envs = batch_report.get("recommended_parallel_envs", 4)

    if batch_size and batch_size < 256:
        recommendations.append(f"Increase gradient accumulation because optimal batch size is only {batch_size}.")
        snippets.append("gradient_accumulation_steps = 4")
    else:
        recommendations.append(f"Use batch size near {batch_size} for high throughput.")

    if sps and sps < 1000:
        recommendations.append("Training throughput is below 1000 samples/s; enable AMP and raise batch size.")
    else:
        recommendations.append("Throughput target is met or close; prioritize thermal stability.")

    if envs > 0:
        recommendations.append(f"Set parallel environments around {envs} for balanced CPU/GPU usage.")

    failed_tests = [r for r in gpu_test_report.get("results", []) if not r.get("passed", False)]
    for item in failed_tests:
        recommendations.append(f"Fix failed test: {item.get('name')} -> {item.get('message')}")

    recommendations.append("Enable TF32 and cudnn benchmark for speed on fixed shapes.")
    recommendations.append("Reserve at least 2.5 GB VRAM for simulation while training.")

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "recommendations": recommendations,
        "code_snippets": snippets,
        "expected_improvements": {
            "throughput_percent": 10 if sps < 1000 else 5,
            "memory_savings_percent": 25,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate optimization recommendations")
    parser.add_argument("--batch-report", type=Path, default=Path("logs/batch_size_recommendation.json"))
    parser.add_argument("--gpu-test-report", type=Path, default=Path("logs/gpu_test_report.json"))
    parser.add_argument("--parallel-report", type=Path, default=Path("logs/parallel_env_recommendation.json"))
    parser.add_argument("--output", type=Path, default=Path("logs/optimization_advice.json"))
    args = parser.parse_args()

    advice = advise(read_json(args.batch_report), read_json(args.gpu_test_report), read_json(args.parallel_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(advice, indent=2), encoding="utf-8")
    print(json.dumps(advice, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
