#!/usr/bin/env python3
"""Assetto Corsa integration connector."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from utils.config_manager import ConfigError, ConfigManager
from utils.shared_memory_reader import SharedMemoryReader, SharedMemoryUnavailableError

LOGGER = logging.getLogger(__name__)


class AssettoCorsaConnectorError(RuntimeError):
    """Raised for Assetto Corsa connectivity issues."""


class AssettoCorsa_Connector:
    """Manage Assetto Corsa discovery, validation, launch, and telemetry access.

    The connector encapsulates all game-side integration details so the gym
    environment can focus on RL semantics.
    """

    COMMON_PATHS_WINDOWS = [
        Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa"),
        Path("C:/Program Files/Steam/steamapps/common/assettocorsa"),
        Path.home() / "AppData/Local/Steam/steamapps/common/assettocorsa",
    ]
    COMMON_PATHS_LINUX = [
        Path.home() / ".steam/steam/steamapps/common/assettocorsa",
        Path.home() / ".local/share/Steam/steamapps/common/assettocorsa",
    ]

    def __init__(self, config_path: str | Path = "configs/config.json") -> None:
        """Initialize connector with config-backed settings.

        Args:
            config_path: Path to the connector config JSON file.
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        self.install_path = self._resolve_install_path()
        self.shared_memory = SharedMemoryReader(self.config)
        self._last_action: dict[str, float] = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "gear_delta": 0.0,
        }

    def _resolve_install_path(self) -> Path:
        """Resolve Assetto Corsa installation path from config or common locations.

        Returns:
            A valid install path.

        Raises:
            AssettoCorsaConnectorError: If installation path is not found.
        """
        cfg_path = self.config.get("assetto_corsa", {}).get("install_path", "")
        if cfg_path:
            candidate = Path(cfg_path).expanduser()
            if candidate.exists():
                LOGGER.info("Using configured Assetto Corsa path: %s", candidate)
                return candidate

        search_paths = (
            self.COMMON_PATHS_WINDOWS if os.name == "nt" else self.COMMON_PATHS_LINUX
        )
        for candidate in search_paths:
            if candidate.exists():
                LOGGER.info("Discovered Assetto Corsa path: %s", candidate)
                self.config["assetto_corsa"]["install_path"] = str(candidate)
                self.config_manager.save()
                return candidate

        raise AssettoCorsaConnectorError(
            "Assetto Corsa installation not found. Update configs/config.json with assetto_corsa.install_path."
        )

    def validate_installation(self) -> bool:
        """Validate required game files and directories.

        Returns:
            True if installation appears valid.

        Raises:
            AssettoCorsaConnectorError: If required files are missing.
        """
        required = self.config["assetto_corsa"].get("required_files", [])
        missing: list[str] = []
        for relative_path in required:
            if not (self.install_path / relative_path).exists():
                missing.append(relative_path)

        if missing:
            raise AssettoCorsaConnectorError(
                f"Assetto Corsa installation is incomplete. Missing: {', '.join(missing)}"
            )

        LOGGER.info("Assetto Corsa installation validated at %s", self.install_path)
        return True

    def launch_game(
        self, wait_seconds: int = 8, timeout_seconds: int = 120
    ) -> subprocess.Popen[Any]:
        """Launch Assetto Corsa executable and wait briefly for startup.

        Args:
            wait_seconds: Delay to allow game process initialization.
            timeout_seconds: Reserved for future launch-health timeout handling.

        Returns:
            subprocess handle for the game process.

        Raises:
            AssettoCorsaConnectorError: If launch fails.
        """
        del timeout_seconds  # Reserved for future process health checks.

        exe_name = self.config["assetto_corsa"].get(
            "executable_name", "AssettoCorsa.exe"
        )
        exe_path = self.install_path / exe_name
        if not exe_path.exists():
            raise AssettoCorsaConnectorError(f"Game executable not found: {exe_path}")

        try:
            process = subprocess.Popen(
                [str(exe_path)],
                cwd=str(self.install_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            LOGGER.info("Launched Assetto Corsa. PID=%s", process.pid)
            time.sleep(max(0, wait_seconds))
            return process
        except OSError as exc:
            raise AssettoCorsaConnectorError(f"Failed to launch game: {exc}") from exc

    def connect_shared_memory(self) -> None:
        """Start shared memory polling.

        Raises:
            AssettoCorsaConnectorError: If shared memory is unavailable.
        """
        try:
            self.shared_memory.start()
        except SharedMemoryUnavailableError as exc:
            raise AssettoCorsaConnectorError(str(exc)) from exc

    def disconnect_shared_memory(self) -> None:
        """Stop shared memory polling and release resources."""
        self.shared_memory.stop()

    def read_state(self) -> dict[str, Any]:
        """Read latest state in normalized dict format for environment consumption."""
        frame = self.shared_memory.read_latest()
        return {
            "timestamp": frame.timestamp,
            "position": frame.position,
            "velocity": frame.velocity,
            "acceleration": frame.acceleration,
            "forces": frame.forces,
            "orientation": frame.orientation,
            "angular_velocity": frame.angular_velocity,
            "steering": frame.steering,
            "throttle": frame.throttle,
            "brake": frame.brake,
            "gear": frame.gear,
            "rpm": frame.rpm,
            "fuel": frame.fuel,
            "speed_kmh": frame.speed_kmh,
            "track_center_distance": frame.track_center_distance,
            "distance_along_track": frame.distance_along_track,
            "collision": frame.collision,
            "off_track": frame.off_track,
            "damage": frame.damage,
        }

    def send_action(
        self, steering: float, throttle: float, brake: float, gear_delta: float
    ) -> None:
        """Send control action to game input bridge.

        In this phase, actions are cached and logged. In future phases, replace this
        stub with a real game input bridge (vJoy/virtual gamepad/UDP plugin).
        """
        self._last_action = {
            "steering": float(max(-1.0, min(1.0, steering))),
            "throttle": float(max(0.0, min(1.0, throttle))),
            "brake": float(max(0.0, min(1.0, brake))),
            "gear_delta": float(max(-1.0, min(1.0, gear_delta))),
        }
        LOGGER.debug("Action cached for bridge dispatch: %s", self._last_action)

    def last_action(self) -> dict[str, float]:
        """Return latest action command for debugging/inspection."""
        return dict(self._last_action)

    def save_runtime_config(self) -> None:
        """Persist in-memory configuration updates."""
        self.config_manager.validate(self.config)
        self.config_manager._config = self.config
        self.config_manager.save()

    @classmethod
    def from_config(
        cls, config_path: str | Path = "configs/config.json"
    ) -> "AssettoCorsa_Connector":
        """Factory helper for explicit construction patterns."""
        return cls(config_path=config_path)
