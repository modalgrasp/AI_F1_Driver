#!/usr/bin/env python3
"""Gymnasium wrapper for Assetto Corsa F1 reinforcement learning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.assetto_corsa_connector import (
    AssettoCorsa_Connector,
    AssettoCorsaConnectorError,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Track key episode-level metrics."""

    episode_index: int = 0
    steps: int = 0
    total_reward: float = 0.0
    start_time: float = 0.0
    off_track_steps: int = 0
    collisions: int = 0


class F1RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Custom Gymnasium environment for F1 driving in Assetto Corsa.

    Supports both action modes:
    - continuous: Box(4,) [steering, throttle, brake, gear_delta]
    - discrete: MultiDiscrete([5, 3, 3, 3]) mapped to continuous controls
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        config_path: str = "configs/config.json",
        action_mode: str = "continuous",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.connector = AssettoCorsa_Connector(config_path=config_path)
        self.config = self.connector.config

        self.max_episode_steps = int(self.config["training"]["max_episode_steps"])
        self.thresholds = self.config["training"]["termination_thresholds"]
        self.rewards_cfg = self.config["training"]["reward_weights"]

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -1e6,
                    -1e6,
                    -1e6,
                    -400.0,
                    -400.0,
                    -400.0,
                    -np.pi,
                    -np.pi,
                    -np.pi,
                    -30.0,
                    -30.0,
                    -30.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    -100.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1e6,
                    1e6,
                    1e6,
                    400.0,
                    400.0,
                    400.0,
                    np.pi,
                    np.pi,
                    np.pi,
                    30.0,
                    30.0,
                    30.0,
                    1.0,
                    1.0,
                    1.0,
                    8.0,
                    20000.0,
                    450.0,
                    100.0,
                    100000.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        if action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif action_mode == "discrete":
            # steering bins: [-1,-0.5,0,0.5,1], throttle 3 bins, brake 3 bins, gear {-1,0,1}
            self.action_space = spaces.MultiDiscrete([5, 3, 3, 3])
        else:
            raise ValueError("action_mode must be 'continuous' or 'discrete'")

        self.stats = EpisodeStats()
        self._last_distance = 0.0
        self._elapsed_seconds = 0.0
        self._rng = np.random.default_rng(int(self.config["training"].get("seed", 42)))

        try:
            self.connector.validate_installation()
            self.connector.connect_shared_memory()
        except AssettoCorsaConnectorError as exc:
            LOGGER.warning(
                "Live AC connection not available. Falling back to safe telemetry: %s",
                exc,
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to episode start state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        del options
        self.stats.episode_index += 1
        self.stats.steps = 0
        self.stats.total_reward = 0.0
        self.stats.start_time = time.perf_counter()
        self.stats.off_track_steps = 0
        self.stats.collisions = 0
        self._elapsed_seconds = 0.0

        state = self.connector.read_state()
        self._last_distance = float(state["distance_along_track"])
        obs = self._state_to_observation(state)

        info = {"episode": self.stats.episode_index}
        LOGGER.info("Episode %d reset complete.", self.stats.episode_index)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one control step with reward/termination evaluation."""
        t0 = time.perf_counter()
        steer, throttle, brake, gear_delta = self._decode_action(action)
        self.connector.send_action(steer, throttle, brake, gear_delta)

        state = self.connector.read_state()
        obs = self._state_to_observation(state)

        reward = self._compute_reward(state)
        terminated, terminal_reason = self._check_termination(state)
        truncated = self.stats.steps >= self.max_episode_steps

        self.stats.steps += 1
        self.stats.total_reward += reward
        self._elapsed_seconds = time.perf_counter() - self.stats.start_time

        info = {
            "terminal_reason": terminal_reason,
            "episode_steps": self.stats.steps,
            "episode_reward": self.stats.total_reward,
            "elapsed_seconds": self._elapsed_seconds,
            "step_duration_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        """Render diagnostics for current state."""
        if self.render_mode == "human":
            state = self.connector.read_state()
            LOGGER.info(
                "Render | speed=%.1f km/h gear=%d rpm=%.0f progress=%.1f",
                state["speed_kmh"],
                state["gear"],
                state["rpm"],
                state["distance_along_track"],
            )

    def close(self) -> None:
        """Release game telemetry resources."""
        self.connector.disconnect_shared_memory()

    def _decode_action(self, action: np.ndarray) -> tuple[float, float, float, float]:
        """Decode action array from selected control mode."""
        if self.action_mode == "continuous":
            vec = np.asarray(action, dtype=np.float32).flatten()
            if vec.shape[0] != 4:
                raise ValueError("Continuous action must have shape (4,)")
            steer = float(np.clip(vec[0], -1.0, 1.0))
            throttle = float(np.clip(vec[1], 0.0, 1.0))
            brake = float(np.clip(vec[2], 0.0, 1.0))
            gear_delta = float(np.clip(vec[3], -1.0, 1.0))
            return steer, throttle, brake, gear_delta

        discrete = np.asarray(action, dtype=np.int32).flatten()
        if discrete.shape[0] != 4:
            raise ValueError("Discrete action must have shape (4,)")

        steer_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        pedal_values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        gear_values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        steer = float(steer_values[int(np.clip(discrete[0], 0, 4))])
        throttle = float(pedal_values[int(np.clip(discrete[1], 0, 2))])
        brake = float(pedal_values[int(np.clip(discrete[2], 0, 2))])
        gear_delta = float(gear_values[int(np.clip(discrete[3], 0, 2))])
        return steer, throttle, brake, gear_delta

    def _state_to_observation(self, state: dict[str, Any]) -> np.ndarray:
        """Convert telemetry dict into fixed-length observation vector."""
        obs = np.array(
            [
                *state["position"],
                *state["velocity"],
                *state["orientation"],
                *state["angular_velocity"],
                state["steering"],
                state["throttle"],
                state["brake"],
                state["gear"],
                state["rpm"],
                state["speed_kmh"],
                state["track_center_distance"],
                state["distance_along_track"],
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_reward(self, state: dict[str, Any]) -> float:
        """Compute reward from progress, stability, and safety factors."""
        delta_progress = float(state["distance_along_track"]) - self._last_distance
        self._last_distance = float(state["distance_along_track"])

        reward = self.rewards_cfg["forward_progress"] * max(0.0, delta_progress)

        if bool(state["off_track"]):
            reward += self.rewards_cfg["off_track_penalty"]
            self.stats.off_track_steps += 1

        if bool(state["collision"]):
            reward += self.rewards_cfg["collision_penalty"]
            self.stats.collisions += 1

        if float(state["speed_kmh"]) < float(self.thresholds["min_speed_kmh"]):
            reward += self.rewards_cfg["low_speed_penalty"]

        # Approximate racing-line bonus from centerline distance.
        center_distance = abs(float(state["track_center_distance"]))
        reward += self.rewards_cfg["racing_line_bonus"] * max(
            0.0, 1.0 - center_distance / 2.0
        )
        return float(reward)

    def _check_termination(self, state: dict[str, Any]) -> tuple[bool, str]:
        """Evaluate terminal conditions and return reason string."""
        if bool(state["collision"]) and float(state["damage"]) >= float(
            self.thresholds["collision_damage_threshold"]
        ):
            return True, "collision"

        off_track_seconds = self.stats.off_track_steps / max(
            1.0, self.config["physics"]["simulation_rate_hz"]
        )
        if off_track_seconds >= float(self.thresholds["off_track_seconds"]):
            return True, "off_track_timeout"

        if self._elapsed_seconds >= float(self.thresholds["time_limit_seconds"]):
            return True, "time_limit"

        if state["distance_along_track"] >= 5200.0:  # Yas Marina ~5.2km
            return True, "lap_complete"

        return False, "running"


def make_vectorized_env(
    num_envs: int,
    config_path: str = "configs/config.json",
    action_mode: str = "continuous",
) -> gym.vector.SyncVectorEnv:
    """Create a vectorized Gymnasium environment for parallel rollouts.

    Args:
        num_envs: Number of parallel environment instances.
        config_path: Path to JSON configuration.
        action_mode: Either "continuous" or "discrete".

    Returns:
        SyncVectorEnv instance ready for batched interaction.
    """

    def _make_single_env() -> F1RacingEnv:
        return F1RacingEnv(config_path=config_path, action_mode=action_mode)

    env_fns = [_make_single_env for _ in range(max(1, int(num_envs)))]
    return gym.vector.SyncVectorEnv(env_fns)
