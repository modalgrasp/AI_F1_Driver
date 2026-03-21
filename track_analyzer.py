#!/usr/bin/env python3
"""Track analytics module for racing line segmentation and speed modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


def load_track_data(track_id: str) -> dict[str, Any]:
    path = Path("data/tracks") / track_id / "extracted" / f"{track_id}_track_data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_curvature(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dz = np.gradient(z)
    ddx = np.gradient(dx)
    ddz = np.gradient(dz)
    curvature = np.abs(dx * ddz - dz * ddx) / np.power(dx * dx + dz * dz + 1e-9, 1.5)
    return curvature


def classify_segments(radius: np.ndarray) -> list[str]:
    labels: list[str] = []
    for r in radius:
        if r < 50.0:
            labels.append("slow_corner")
        elif r < 150.0:
            labels.append("medium_corner")
        elif r > 300.0:
            labels.append("straight")
        else:
            labels.append("fast_corner")
    return labels


def theoretical_speed_kmh(
    curvature: np.ndarray, grip: float = 1.9, downforce_factor: float = 1.25
) -> np.ndarray:
    """Compute simple lateral-limit based speed estimate.

    v = sqrt(a_lat / curvature) where a_lat approx grip * g * downforce_factor.
    """
    g = 9.81
    a_lat = grip * g * downforce_factor
    speed_mps = np.sqrt(a_lat / np.maximum(curvature, 1e-5))
    speed_kmh = np.clip(speed_mps * 3.6, 40.0, 350.0)
    return speed_kmh


def detect_braking_accel_zones(speed_kmh: np.ndarray) -> tuple[list[int], list[int]]:
    speed_mps = speed_kmh / 3.6
    accel = np.gradient(speed_mps)
    braking_idx = np.where(accel < np.percentile(accel, 10))[0].tolist()
    accel_idx = np.where(accel > np.percentile(accel, 90))[0].tolist()
    return braking_idx, accel_idx


def gear_hint(speed_kmh: np.ndarray) -> np.ndarray:
    bins = np.array([0, 70, 110, 145, 180, 220, 260, 300, 380], dtype=np.float64)
    gears = np.digitize(speed_kmh, bins)
    return np.clip(gears, 1, 8)


def sector_estimates(
    distance_m: np.ndarray, speed_kmh: np.ndarray, sectors: np.ndarray
) -> dict[int, float]:
    speed_mps = np.maximum(speed_kmh / 3.6, 1.0)
    ds = np.diff(distance_m, prepend=distance_m[0])
    dt = ds / speed_mps
    results: dict[int, float] = {}
    for sec in sorted(set(sectors.tolist())):
        results[int(sec)] = float(np.sum(dt[sectors == sec]))
    return results


def run_analysis(track_id: str) -> dict[str, Any]:
    payload = load_track_data(track_id)
    wp = payload["waypoints"]
    xyz = np.array([[w["x"], w["y"], w["z"]] for w in wp], dtype=np.float64)
    sectors = np.array([w.get("sector", 1) for w in wp], dtype=np.int32)

    d = np.sqrt(np.sum(np.diff(xyz[:, [0, 2]], axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0.0)

    raw_curvature = compute_curvature(xyz[:, 0], xyz[:, 2])
    window = min(len(raw_curvature) - (len(raw_curvature) + 1) % 2, 41)
    window = max(window, 5)
    smooth_curvature = savgol_filter(
        raw_curvature, window_length=window, polyorder=3, mode="interp"
    )

    radius = 1.0 / np.maximum(smooth_curvature, 1e-5)
    segment_types = classify_segments(radius)

    v_theory = theoretical_speed_kmh(smooth_curvature)
    braking_idx, accel_idx = detect_braking_accel_zones(v_theory)
    gears = gear_hint(v_theory)

    sector_times = sector_estimates(s, v_theory, sectors)
    lap_estimate = float(sum(sector_times.values()))

    analysis = {
        "track_id": track_id,
        "timestamp": payload["metadata"]["extracted_at"],
        "distance_m": s.tolist(),
        "curvature": smooth_curvature.tolist(),
        "radius_m": radius.tolist(),
        "segment_type": segment_types,
        "theoretical_speed_kmh": v_theory.tolist(),
        "braking_zone_indices": braking_idx,
        "acceleration_zone_indices": accel_idx,
        "gear_hints": gears.tolist(),
        "sector_time_estimates_s": sector_times,
        "lap_time_estimate_s": lap_estimate,
        "energy_deployment_opportunities": [
            {"index": int(i), "type": "deploy"}
            for i in accel_idx[:: max(1, len(accel_idx) // 12)]
        ],
        "energy_harvest_zones": [
            {"index": int(i), "type": "harvest"}
            for i in braking_idx[:: max(1, len(braking_idx) // 12)]
        ],
        "benchmark_comparison": {
            "benchmark_lap_s": 82.109,
            "delta_s": lap_estimate - 82.109,
        },
    }

    out_dir = Path("data/tracks") / track_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{track_id}_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze extracted racing line and track geometry."
    )
    parser.add_argument("--track-id", default="yas_marina")
    args = parser.parse_args()
    result = run_analysis(args.track_id)
    print(f"Analysis complete. Lap estimate: {result['lap_time_estimate_s']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
