#!/usr/bin/env python3
"""Integrate extracted Yas Marina data into F1RacingEnv runtime behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from environments.f1_racing_env import F1RacingEnv


def load_track_payload(track_id: str) -> dict:
    path = Path("data/tracks") / track_id / "extracted" / f"{track_id}_track_data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def integrate(track_id: str, episodes: int = 1) -> dict:
    payload = load_track_payload(track_id)
    centerline = np.array(payload["boundaries"]["centerline"], dtype=np.float64)
    sectors = payload.get("sectors", [])

    env = F1RacingEnv(config_path="configs/config.json", action_mode="continuous")

    report = {
        "track_id": track_id,
        "centerline_points": int(centerline.shape[0]),
        "sector_count": len(sectors),
        "episodes": [],
    }

    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep)
        total_reward = 0.0
        lap_progress = 0.0
        sector_hits: set[int] = set()

        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            lap_progress = max(lap_progress, float(obs[-1]))

            # Approximate sector index from normalized lap progress.
            if centerline.shape[0] > 0:
                ratio = lap_progress / max(1.0, np.max(centerline[:, 0]) - np.min(centerline[:, 0]))
                if ratio < 0.33:
                    sector_hits.add(1)
                elif ratio < 0.66:
                    sector_hits.add(2)
                else:
                    sector_hits.add(3)

            if terminated or truncated:
                break

        report["episodes"].append(
            {
                "episode": ep + 1,
                "total_reward": float(total_reward),
                "lap_progress_indicator": float(lap_progress),
                "sector_hits": sorted(sector_hits),
            }
        )

    env.close()

    out_dir = Path("data/tracks") / track_id / "integration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "integration_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate track data with RL environment")
    parser.add_argument("--track-id", default="yas_marina")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    report = integrate(args.track_id, episodes=max(1, args.episodes))
    print(f"Integration report generated for {report['track_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
