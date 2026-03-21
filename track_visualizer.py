#!/usr/bin/env python3
"""Visualize extracted track geometry in 2D/3D and analysis charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def load_payload(track_id: str) -> dict:
    path = Path("data/tracks") / track_id / "extracted" / f"{track_id}_track_data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plot_2d(track_id: str, payload: dict, out_dir: Path) -> None:
    center = np.array(payload["boundaries"]["centerline"], dtype=np.float64)
    left = np.array(payload["boundaries"]["left"], dtype=np.float64)
    right = np.array(payload["boundaries"]["right"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        center[:, 0], center[:, 2], label="Racing Line", color="tab:blue", linewidth=2
    )
    ax.plot(left[:, 0], left[:, 2], label="Left Boundary", color="tab:green", alpha=0.7)
    ax.plot(
        right[:, 0], right[:, 2], label="Right Boundary", color="tab:red", alpha=0.7
    )

    for turn in payload.get("turns", [])[:16]:
        apex = np.array(turn["apex_xyz"], dtype=np.float64)
        ax.scatter(apex[0], apex[2], color="black", s=20)
        ax.text(apex[0], apex[2], f"T{turn['turn']}", fontsize=8)

    drs = payload.get("drs_zones", [])
    if len(drs) >= 1:
        ax.text(center[0, 0], center[0, 2], "DRS zones configured", color="purple")

    ax.set_title(f"{track_id} Layout (2D Overhead)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{track_id}_layout_2d.png", dpi=200)
    fig.savefig(out_dir / f"{track_id}_layout_2d.svg")
    plt.close(fig)


def plot_3d(track_id: str, payload: dict, out_dir: Path) -> None:
    center = np.array(payload["boundaries"]["centerline"], dtype=np.float64)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=center[:, 0],
            y=center[:, 2],
            z=center[:, 1],
            mode="lines",
            name="Racing Line",
            line={"width": 6, "color": "royalblue"},
        )
    )
    fig.update_layout(
        title=f"{track_id} 3D Racing Line",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Z (m)",
            "zaxis_title": "Y Elevation (m)",
        },
    )
    fig.write_html(out_dir / f"{track_id}_layout_3d.html")


def plot_curvature_speed(track_id: str, payload: dict, out_dir: Path) -> None:
    waypoints = payload["waypoints"]
    xyz = np.array([[w["x"], w["y"], w["z"]] for w in waypoints], dtype=np.float64)
    speed = np.array([w["speed_hint_kmh"] for w in waypoints], dtype=np.float64)

    d = np.sqrt(np.sum(np.diff(xyz[:, [0, 2]], axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0.0)
    x = xyz[:, 0]
    z = xyz[:, 2]
    dx = np.gradient(x)
    dz = np.gradient(z)
    ddx = np.gradient(dx)
    ddz = np.gradient(dz)
    curvature = np.abs(dx * ddz - dz * ddx) / np.power(dx * dx + dz * dz + 1e-9, 1.5)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(s, curvature, color="tab:orange")
    axes[0].set_ylabel("Curvature (1/m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s, speed, color="tab:blue")
    axes[1].set_ylabel("Speed hint (km/h)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(s, xyz[:, 1], color="tab:green")
    axes[2].set_ylabel("Elevation (m)")
    axes[2].set_xlabel("Distance along track (m)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{track_id} Curvature / Speed / Elevation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{track_id}_curvature_speed_elevation.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize extracted track data")
    parser.add_argument("--track-id", default="yas_marina")
    args = parser.parse_args()

    payload = load_payload(args.track_id)
    out_dir = Path("data/tracks") / args.track_id / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_2d(args.track_id, payload, out_dir)
    plot_3d(args.track_id, payload, out_dir)
    plot_curvature_speed(args.track_id, payload, out_dir)
    print(f"Visualizations generated in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
