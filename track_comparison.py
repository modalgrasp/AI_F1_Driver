#!/usr/bin/env python3
"""Compare multiple Yas Marina track extraction outputs and recommend best candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_track(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    meta = payload.get("metadata", {})
    turns = payload.get("turns", [])
    surfaces = payload.get("surfaces", [])
    boundaries = payload.get("boundaries", {})

    score_parts: dict[str, float] = {}
    score_parts["waypoint_density"] = min(meta.get("waypoint_count", 0) / 2000.0, 1.0)
    score_parts["turn_coverage"] = min(len(turns) / 16.0, 1.0)
    score_parts["surface_defs"] = min(len(surfaces) / 8.0, 1.0)
    score_parts["boundary_presence"] = (
        1.0 if boundaries.get("left") and boundaries.get("right") else 0.0
    )

    total = float(np.mean(list(score_parts.values())))
    return total, score_parts


def compare(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        payload = load_payload(path)
        score, parts = score_track(payload)
        rows.append(
            {
                "file": str(path),
                "track_id": payload.get("metadata", {}).get("track_id", path.stem),
                "score": score,
                "score_breakdown": parts,
                "waypoints": payload.get("metadata", {}).get("waypoint_count", 0),
                "turns": len(payload.get("turns", [])),
                "surface_defs": len(payload.get("surfaces", [])),
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    recommendation = rows[0] if rows else None
    return {
        "candidates": rows,
        "recommended": recommendation,
        "notes": [
            "Higher score indicates richer usable geometry/metadata.",
            "Manually verify visual fidelity in-game before final selection.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare extracted track variants")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Paths to *_track_data.json files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tracks/yas_marina/comparison/track_comparison_report.json"),
    )
    args = parser.parse_args()

    report = compare(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison report saved: {args.output}")
    if report["recommended"]:
        print(f"Recommended: {report['recommended']['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
