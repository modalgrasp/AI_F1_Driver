#!/usr/bin/env python3
"""Track testing suite: automated checks + manual checklist report."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from environments.f1_racing_env import F1RacingEnv
from track_validator import TrackValidator


def run_automated_tests(track_id: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    validator = TrackValidator(track_id=track_id)
    report = validator.run()
    results.append(
        {
            "test": "track_validation",
            "pass": report["success"],
            "detail": "validator executed",
        }
    )

    env = F1RacingEnv(config_path="configs/config.json", action_mode="continuous")
    obs, _ = env.reset(seed=123)
    results.append(
        {"test": "env_reset", "pass": len(obs) == 20, "detail": "observation shape"}
    )

    step_times = []
    for _ in range(200):
        action = env.action_space.sample()
        s0 = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        step_times.append((time.perf_counter() - s0) * 1000.0)
        if terminated or truncated:
            env.reset()

    env.close()
    load_time = (time.perf_counter() - t0) * 1000.0
    avg_step_ms = sum(step_times) / max(1, len(step_times))

    results.append(
        {
            "test": "performance_loading_ms",
            "pass": load_time < 15000.0,
            "detail": load_time,
        }
    )
    results.append(
        {
            "test": "performance_step_ms",
            "pass": avg_step_ms < 5.0,
            "detail": avg_step_ms,
        }
    )

    return {
        "track_id": track_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "automated_results": results,
        "metrics": {
            "loading_time_ms": load_time,
            "average_step_ms": avg_step_ms,
            "estimated_render_fps": 1000.0 / max(1e-6, avg_step_ms),
        },
        "manual_procedure": [
            "Open Assetto Corsa and verify track appears in content list.",
            "Start practice session on Yas Marina and confirm lap timer increments.",
            "Run AI opponents and verify they complete laps without major crashes.",
            "Enter and exit pit lane to verify speed limits and pit lane spline.",
            "Capture screenshot/video of map and sector timing overlays.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run track test suite")
    parser.add_argument("--track-id", default="yas_marina")
    args = parser.parse_args()

    report = run_automated_tests(args.track_id)
    out_dir = Path("data/tracks") / args.track_id / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "track_test_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Track test report: {out_file}")
    failed = [r for r in report["automated_results"] if not r["pass"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
