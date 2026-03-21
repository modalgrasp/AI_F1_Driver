#!/usr/bin/env python3
"""Integration test script for F1RacingEnv."""

from __future__ import annotations

import logging
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.f1_racing_env import F1RacingEnv
from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


def run_random_policy_test(episodes: int = 5) -> None:
    """Run random agent episodes and assert basic environment correctness."""
    env = F1RacingEnv(
        config_path="configs/config.json", action_mode="continuous", render_mode="none"
    )
    assert env.observation_space.shape == (20,), "Unexpected observation space shape"

    tracemalloc.start()
    memory_snapshots: list[int] = []

    step_times: list[float] = []
    for episode in range(episodes):
        observation, info = env.reset(seed=episode)
        assert env.observation_space.contains(
            observation
        ), "Reset observation out of bounds"
        LOGGER.info("Episode %d started. info=%s", episode + 1, info)

        total_reward = 0.0
        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < 1000:
            action = env.action_space.sample()
            if hasattr(env.action_space, "contains"):
                assert env.action_space.contains(action), "Sampled action out of bounds"

            t0 = time.perf_counter()
            next_observation, reward, terminated, truncated, step_info = env.step(
                action
            )
            step_times.append((time.perf_counter() - t0) * 1000.0)

            assert env.observation_space.contains(
                next_observation
            ), "Step observation out of bounds"
            assert isinstance(reward, float), "Reward must be float"
            assert isinstance(step_info, dict), "Info must be dict"

            observation = next_observation
            total_reward += reward
            step_count += 1

        current, peak = tracemalloc.get_traced_memory()
        memory_snapshots.append(current)

        LOGGER.info(
            "Episode %d finished | reward=%.3f steps=%d terminated=%s truncated=%s peak_mem_kb=%.1f",
            episode + 1,
            total_reward,
            step_count,
            terminated,
            truncated,
            peak / 1024.0,
        )

    env.render()
    env.close()

    avg_step_ms = float(np.mean(step_times)) if step_times else float("nan")
    LOGGER.info("Average step duration: %.4f ms", avg_step_ms)

    if len(memory_snapshots) >= 2:
        delta = memory_snapshots[-1] - memory_snapshots[0]
        LOGGER.info("Memory delta after %d episodes: %d bytes", episodes, delta)
        assert delta < 50 * 1024 * 1024, "Potential memory leak detected (>50MB growth)"

    assert avg_step_ms < 10.0, "Step function performance regression (>10ms average)"
    tracemalloc.stop()


def main() -> int:
    """Entrypoint for script execution."""
    setup_logging(Path("logs"), level="INFO", console=True)
    LOGGER.info("Starting Gymnasium API compliance smoke test")

    run_random_policy_test(episodes=5)
    LOGGER.info("Environment test completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
