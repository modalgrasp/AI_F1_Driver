#!/usr/bin/env python3
"""Generate track-specific RL configuration for Yas Marina."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.config_manager import ConfigManager


def generate_track_config(track_id: str, base_config_path: Path, output_path: Path) -> dict[str, Any]:
    base = ConfigManager(base_config_path).load()

    track_cfg: dict[str, Any] = {
        "track_id": track_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "observation_space_extensions": {
            "include_distance_to_apex": True,
            "include_curvature_lookahead": True,
            "include_sector_id": True,
            "lookahead_meters": 120.0,
        },
        "reward_shaping": {
            "global": {
                "progress_weight": 2.5,
                "off_track_penalty": -4.0,
                "collision_penalty": -10.0,
                "centerline_bonus": 1.2,
            },
            "turn_specific": {
                "T1": {"entry_stability_bonus": 0.4},
                "T7": {"braking_precision_bonus": 0.5},
                "T16": {"exit_speed_bonus": 0.6},
            },
            "sector_bonuses": {
                "sector_1_clean": 1.0,
                "sector_2_top_speed": 1.2,
                "sector_3_consistency": 1.0,
            },
        },
        "initial_state_distribution": {
            "spawn_mode": "random_along_track",
            "distance_range_m": [0.0, 5281.0],
            "speed_range_kmh": [90.0, 260.0],
            "yaw_noise_deg": 2.0,
        },
        "termination_conditions": {
            "lap_complete": True,
            "off_track_time_limit_s": 2.5,
            "collision_immediate_terminate": True,
            "stuck_speed_threshold_kmh": 5.0,
            "stuck_duration_s": 8.0,
        },
        "track_limits": {
            "default_half_width_m": 6.0,
            "kerb_allowance_m": 0.6,
            "pit_lane_speed_limit_kmh_practice_qualifying": 80,
            "pit_lane_speed_limit_kmh_race": 60,
        },
        "rendering": {
            "show_racing_line": True,
            "show_boundaries": True,
            "camera_presets": [
                {"name": "start_finish", "xyz": [0.0, 45.0, 0.0]},
                {"name": "sector2_straight", "xyz": [1300.0, 50.0, 500.0]},
            ],
        },
        "derived_from": {
            "base_config": str(base_config_path),
            "base_track": base.get("race", {}).get("track", "unknown"),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(track_cfg, indent=2), encoding="utf-8")
    return track_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Yas Marina RL config")
    parser.add_argument("--track-id", default="yas_marina")
    parser.add_argument("--base-config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/yas_marina_config.json"))
    args = parser.parse_args()

    cfg = generate_track_config(args.track_id, args.base_config, args.output)
    print(f"Generated config: {args.output}")
    print(f"Reward keys: {list(cfg['reward_shaping']['global'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
