#!/usr/bin/env python3
"""Configuration manager for the F1 autonomous racing project."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class ConfigManager:
    """Load, validate, update, and persist project configuration.

    Attributes:
        config_path: Path to the JSON configuration file.
        _config: In-memory configuration dictionary.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "assetto_corsa": {
            "install_path": "",
            "executable_name": "AssettoCorsa.exe",
            "steam_app_id": 244210,
            "required_files": [
                "AssettoCorsa.exe",
                "acs.exe",
                "content/tracks",
                "content/cars",
            ],
            "shared_memory": {
                "enabled": True,
                "polling_hz": 100,
                "segment_names": {
                    "local": "Local",
                    "physics": "Physics",
                    "graphics": "Graphics",
                    "static": "Static",
                },
                "legacy_segment_names": {
                    "physics": "Local\\acpmf_physics",
                    "graphics": "Local\\acpmf_graphics",
                    "static": "Local\\acpmf_static",
                },
                "use_double_buffer": True,
                "read_timeout_ms": 50,
            },
        },
        "race": {
            "track": "yas_marina",
            "car": "ks_ferrari_sf70h",
            "session_type": "practice",
            "max_laps": 1,
        },
        "graphics": {
            "width": 1920,
            "height": 1080,
            "quality": "high",
            "vsync": False,
        },
        "physics": {
            "timestep_seconds": 0.01,
            "simulation_rate_hz": 100,
            "action_repeat": 1,
        },
        "training": {
            "seed": 42,
            "max_episode_steps": 5000,
            "reward_weights": {
                "forward_progress": 2.0,
                "off_track_penalty": -3.0,
                "collision_penalty": -8.0,
                "low_speed_penalty": -0.5,
                "racing_line_bonus": 1.0,
            },
            "termination_thresholds": {
                "off_track_seconds": 3.0,
                "collision_damage_threshold": 0.4,
                "min_speed_kmh": 10.0,
                "time_limit_seconds": 300,
            },
        },
        "logging": {
            "level": "INFO",
            "directory": "logs",
            "console": True,
        },
    }

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load config file and merge defaults.

        Returns:
            Final validated configuration dictionary.

        Raises:
            ConfigError: If the file is invalid JSON or contains invalid values.
        """
        if not self.config_path.exists():
            LOGGER.warning("Config not found at %s. Creating with defaults.", self.config_path)
            self._config = deepcopy(self.DEFAULT_CONFIG)
            self.save()
            return self._config

        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                user_config = json.load(file)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {self.config_path}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Failed to read config {self.config_path}: {exc}") from exc

        self._config = self._merge_dicts(deepcopy(self.DEFAULT_CONFIG), user_config)
        self.validate(self._config)
        return self._config

    def get(self) -> dict[str, Any]:
        """Return loaded config, loading it first if needed."""
        if not self._config:
            return self.load()
        return self._config

    def update(self, dotted_key: str, value: Any) -> None:
        """Update a configuration field at runtime.

        Args:
            dotted_key: Dot-separated key path (example: "training.max_episode_steps").
            value: New value to set.

        Raises:
            ConfigError: If the key path is invalid.
        """
        config = self.get()
        parts = dotted_key.split(".")
        cursor: dict[str, Any] = config
        for key in parts[:-1]:
            next_val = cursor.get(key)
            if not isinstance(next_val, dict):
                raise ConfigError(f"Invalid config key path: {dotted_key}")
            cursor = next_val

        cursor[parts[-1]] = value
        self.validate(config)
        self._config = config

    def save(self) -> None:
        """Persist current configuration to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.get()
        try:
            with self.config_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
        except OSError as exc:
            raise ConfigError(f"Failed to save config {self.config_path}: {exc}") from exc

    @classmethod
    def validate(cls, config: dict[str, Any]) -> None:
        """Validate critical configuration values.

        Args:
            config: Candidate configuration dictionary.

        Raises:
            ConfigError: If required values are missing or invalid.
        """
        try:
            polling_hz = int(config["assetto_corsa"]["shared_memory"]["polling_hz"])
            if polling_hz < 1 or polling_hz > 1000:
                raise ConfigError("assetto_corsa.shared_memory.polling_hz must be in [1, 1000]")

            max_steps = int(config["training"]["max_episode_steps"])
            if max_steps < 1:
                raise ConfigError("training.max_episode_steps must be > 0")

            timestep = float(config["physics"]["timestep_seconds"])
            if timestep <= 0.0:
                raise ConfigError("physics.timestep_seconds must be > 0")

            level = str(config["logging"]["level"]).upper()
            if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                raise ConfigError("logging.level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL")

        except KeyError as exc:
            raise ConfigError(f"Missing required config key: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid config value type: {exc}") from exc

    @staticmethod
    def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = ConfigManager._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
