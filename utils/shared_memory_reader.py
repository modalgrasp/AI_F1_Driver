#!/usr/bin/env python3
"""Assetto Corsa shared memory telemetry reader.

This implementation is intentionally defensive: it gracefully degrades when
shared memory segments are unavailable and exposes a stable dictionary API.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class SharedMemoryFrame:
    """Telemetry frame consumed by the gym environment."""

    timestamp: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    forces: tuple[float, float, float]
    orientation: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    steering: float
    throttle: float
    brake: float
    gear: int
    rpm: float
    fuel: float
    speed_kmh: float
    track_center_distance: float
    distance_along_track: float
    collision: bool
    off_track: bool
    damage: float


class SharedMemoryUnavailableError(RuntimeError):
    """Raised when shared memory could not be opened."""


class SharedMemoryReader:
    """Read Assetto Corsa telemetry from shared memory at configurable polling rate.

    Supports both modern logical segment names (`Local`, `Physics`, `Graphics`, `Static`)
    and legacy names used by some AC integrations (`Local\\acpmf_*`).
    """

    _WINDOWS_DEFAULT_SEGMENTS = {
        "physics": ["Physics", "Local\\acpmf_physics"],
        "graphics": ["Graphics", "Local\\acpmf_graphics"],
        "static": ["Static", "Local\\acpmf_static"],
        "local": ["Local"],
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        shared_cfg = config.get("assetto_corsa", {}).get("shared_memory", {})
        self.polling_hz = int(shared_cfg.get("polling_hz", 100))
        self._poll_interval = 1.0 / max(1, self.polling_hz)
        self._enabled = bool(shared_cfg.get("enabled", True))
        self._timeout_ms = int(shared_cfg.get("read_timeout_ms", 50))
        self._use_double_buffer = bool(shared_cfg.get("use_double_buffer", True))

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cache: deque[SharedMemoryFrame] = deque(
            maxlen=2 if self._use_double_buffer else 1
        )
        self._mappings: dict[str, mmap.mmap] = {}

    def start(self) -> None:
        """Start background polling loop and initialize memory maps."""
        if not self._enabled:
            LOGGER.warning("Shared memory reader is disabled by configuration.")
            return

        self._open_mappings()
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="ac-shmem-poller"
        )
        self._thread.start()
        LOGGER.info("Shared memory polling started at %d Hz.", self.polling_hz)

    def stop(self) -> None:
        """Stop polling and release memory maps."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        for name, mapping in list(self._mappings.items()):
            try:
                mapping.close()
            except Exception:
                LOGGER.exception("Failed closing mapping: %s", name)
        self._mappings.clear()
        LOGGER.info("Shared memory polling stopped.")

    def read_latest(self) -> SharedMemoryFrame:
        """Return latest frame or a safe fallback frame if unavailable."""
        with self._lock:
            if self._cache:
                return self._cache[-1]
        return self._fallback_frame()

    def _open_mappings(self) -> None:
        """Open all available shared memory segments.

        Raises:
            SharedMemoryUnavailableError: If no telemetry segments can be opened.
        """
        if os.name != "nt":
            LOGGER.warning(
                "Native mmap telemetry mapping currently implemented for Windows only."
            )
            return

        opened = 0
        for key, candidates in self._WINDOWS_DEFAULT_SEGMENTS.items():
            for name in candidates:
                try:
                    mapping = mmap.mmap(-1, 4096, tagname=name, access=mmap.ACCESS_READ)
                    self._mappings[key] = mapping
                    opened += 1
                    LOGGER.info(
                        "Opened shared memory segment '%s' for key '%s'.", name, key
                    )
                    break
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

        if opened == 0:
            raise SharedMemoryUnavailableError(
                "No Assetto Corsa shared memory segment could be opened. "
                "Launch the game with shared memory enabled first."
            )

    def _poll_loop(self) -> None:
        """Background polling loop that keeps latest telemetry in a tiny ring buffer."""
        next_tick = time.perf_counter()
        while self._running:
            frame = self._read_single_frame()
            with self._lock:
                self._cache.append(frame)

            next_tick += self._poll_interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If delayed, reset schedule to avoid accumulating lag.
                next_tick = time.perf_counter()

    def _read_single_frame(self) -> SharedMemoryFrame:
        """Read one telemetry frame from shared memory segments."""
        if not self._mappings:
            return self._fallback_frame()

        try:
            physics = self._mappings.get("physics")
            if physics is None:
                return self._fallback_frame()

            physics.seek(0)
            blob = physics.read(128)
            # Placeholder minimal layout: first 9 floats are position+velocity+orientation.
            # Real AC struct is richer and can be mapped incrementally in future phases.
            values = struct.unpack("<9f", blob[:36])
            px, py, pz, vx, vy, vz, roll, pitch, yaw = values

            frame = SharedMemoryFrame(
                timestamp=time.time(),
                position=(px, py, pz),
                velocity=(vx, vy, vz),
                acceleration=(0.0, 0.0, 0.0),
                forces=(0.0, 0.0, 0.0),
                orientation=(roll, pitch, yaw),
                angular_velocity=(0.0, 0.0, 0.0),
                steering=0.0,
                throttle=0.0,
                brake=0.0,
                gear=1,
                rpm=max(0.0, abs(vx) * 100.0),
                fuel=20.0,
                speed_kmh=max(0.0, (vx * vx + vy * vy + vz * vz) ** 0.5 * 3.6),
                track_center_distance=0.0,
                distance_along_track=max(0.0, px),
                collision=False,
                off_track=False,
                damage=0.0,
            )
            return self._sanitize_frame(frame)
        except Exception:
            LOGGER.exception(
                "Shared memory frame read failed. Returning fallback frame."
            )
            return self._fallback_frame()

    def _sanitize_frame(self, frame: SharedMemoryFrame) -> SharedMemoryFrame:
        """Apply sanity checks to avoid unstable values entering RL loop."""
        speed = min(max(frame.speed_kmh, 0.0), 450.0)
        rpm = min(max(frame.rpm, 0.0), 20000.0)
        gear = min(max(frame.gear, -1), 8)
        return SharedMemoryFrame(
            timestamp=frame.timestamp,
            position=tuple(float(v) for v in frame.position),
            velocity=tuple(float(v) for v in frame.velocity),
            acceleration=tuple(float(v) for v in frame.acceleration),
            forces=tuple(float(v) for v in frame.forces),
            orientation=tuple(float(v) for v in frame.orientation),
            angular_velocity=tuple(float(v) for v in frame.angular_velocity),
            steering=float(frame.steering),
            throttle=min(max(float(frame.throttle), 0.0), 1.0),
            brake=min(max(float(frame.brake), 0.0), 1.0),
            gear=int(gear),
            rpm=float(rpm),
            fuel=min(max(float(frame.fuel), 0.0), 200.0),
            speed_kmh=float(speed),
            track_center_distance=float(frame.track_center_distance),
            distance_along_track=float(frame.distance_along_track),
            collision=bool(frame.collision),
            off_track=bool(frame.off_track),
            damage=min(max(float(frame.damage), 0.0), 1.0),
        )

    @staticmethod
    def _fallback_frame() -> SharedMemoryFrame:
        """Fallback frame used when live shared memory is not available."""
        return SharedMemoryFrame(
            timestamp=time.time(),
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            forces=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            steering=0.0,
            throttle=0.0,
            brake=0.0,
            gear=1,
            rpm=0.0,
            fuel=0.0,
            speed_kmh=0.0,
            track_center_distance=0.0,
            distance_along_track=0.0,
            collision=False,
            off_track=False,
            damage=0.0,
        )

    def __enter__(self) -> "SharedMemoryReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.stop()
