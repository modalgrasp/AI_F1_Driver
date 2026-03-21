#!/usr/bin/env python3
"""Prepare racing lines for downstream optimization and RL usage."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


def load_payload(track_id: str) -> dict[str, Any]:
    path = Path("data/tracks") / track_id / "extracted" / f"{track_id}_track_data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resample_line(xyz: np.ndarray, spacing_m: float = 1.0) -> np.ndarray:
    d = np.sqrt(np.sum(np.diff(xyz[:, [0, 2]], axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0.0)
    s_new = np.arange(0.0, s[-1], spacing_m)

    fx = interp1d(s, xyz[:, 0], kind="linear", fill_value="extrapolate")
    fy = interp1d(s, xyz[:, 1], kind="linear", fill_value="extrapolate")
    fz = interp1d(s, xyz[:, 2], kind="linear", fill_value="extrapolate")
    out = np.column_stack([fx(s_new), fy(s_new), fz(s_new)])
    return out


def smooth_line(xyz: np.ndarray) -> np.ndarray:
    window = min(len(xyz) - (len(xyz) + 1) % 2, 31)
    window = max(window, 5)
    smoothed = np.copy(xyz)
    for col in range(3):
        smoothed[:, col] = savgol_filter(
            smoothed[:, col], window_length=window, polyorder=3, mode="interp"
        )
    return smoothed


def tangent_normal(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.roll(xz, -1, axis=0) - xz
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    tangent = d / norm
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def lookahead_features(curvature: np.ndarray, horizon: int = 50) -> np.ndarray:
    features = np.zeros((len(curvature), 2), dtype=np.float32)
    for i in range(len(curvature)):
        j = min(len(curvature), i + horizon)
        segment = curvature[i:j]
        features[i, 0] = float(np.mean(segment))
        features[i, 1] = float(np.max(segment))
    return features


def preprocess(track_id: str) -> dict[str, Any]:
    payload = load_payload(track_id)
    center = np.array(payload["boundaries"]["centerline"], dtype=np.float64)
    ai_line = np.array(
        [[w["x"], w["y"], w["z"]] for w in payload["waypoints"]], dtype=np.float64
    )

    center_uniform = resample_line(center, spacing_m=1.0)
    ai_uniform = resample_line(ai_line, spacing_m=1.0)
    ai_smooth = smooth_line(ai_uniform)

    # Keep memory and save times bounded for large extracted lines.
    max_points = 30000
    if ai_smooth.shape[0] > max_points:
        idx = np.linspace(0, ai_smooth.shape[0] - 1, max_points).astype(np.int64)
        ai_smooth = ai_smooth[idx]
    if center_uniform.shape[0] > max_points:
        idx = np.linspace(0, center_uniform.shape[0] - 1, max_points).astype(np.int64)
        center_uniform = center_uniform[idx]

    tangent, normal = tangent_normal(ai_smooth[:, [0, 2]])

    dx = np.gradient(ai_smooth[:, 0])
    dz = np.gradient(ai_smooth[:, 2])
    ddx = np.gradient(dx)
    ddz = np.gradient(dz)
    curvature = np.abs(dx * ddz - dz * ddx) / np.power(dx * dx + dz * dz + 1e-9, 1.5)
    lookahead = lookahead_features(curvature, horizon=80)

    min_curvature_line = (
        ai_smooth
        - np.column_stack([normal[:, 0], np.zeros(len(normal)), normal[:, 1]]) * 0.5
    )
    kdtree = cKDTree(ai_smooth[:, [0, 2]])
    _, nearest_idx_demo = kdtree.query(ai_smooth[0, [0, 2]], k=5)

    result = {
        "track_id": track_id,
        "spacing_m": 1.0,
        "centerline": center_uniform.tolist(),
        "ai_line": ai_smooth.tolist(),
        "min_curvature_line": min_curvature_line.tolist(),
        "tangent_vectors": tangent.tolist(),
        "normal_vectors": normal.tolist(),
        "curvature": curvature.tolist(),
        "lookahead_features": lookahead.tolist(),
        "spatial_index": {
            "type": "cKDTree",
            "indexed_dimensions": ["x", "z"],
            "point_count": int(ai_smooth.shape[0]),
            "demo_nearest_indices": np.asarray(nearest_idx_demo).astype(int).tolist(),
        },
    }

    out_dir = Path("data/tracks") / track_id / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{track_id}_preprocessed.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez(
        out_dir / f"{track_id}_preprocessed.npz",
        centerline=center_uniform.astype(np.float32),
        ai_line=ai_smooth.astype(np.float32),
        min_curvature_line=min_curvature_line.astype(np.float32),
        tangent=tangent.astype(np.float32),
        normal=normal.astype(np.float32),
        curvature=curvature.astype(np.float32),
        lookahead=lookahead.astype(np.float32),
    )
    with (out_dir / f"{track_id}_preprocessed.pkl").open("wb") as fh:
        pickle.dump(result, fh)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess racing lines for optimization"
    )
    parser.add_argument("--track-id", default="yas_marina")
    args = parser.parse_args()
    result = preprocess(args.track_id)
    print(f"Preprocessed {args.track_id}: {len(result['ai_line'])} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
