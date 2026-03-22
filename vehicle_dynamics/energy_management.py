#!/usr/bin/env python3
"""Energy management logic for F1 2026 hybrid powertrain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DeploymentZone:
    start: float
    end: float
    level: str


@dataclass
class HarvestZone:
    start: float
    end: float
    level: str


class EnergyManagementSystem:
    """Track-aware deployment and harvesting strategy manager."""

    def __init__(self, track_data: dict[str, Any]) -> None:
        self.track_data = track_data
        self.track_length = float(track_data.get("track_length", 5500.0))
        self.deployment_map, self.harvest_map = self.create_deployment_map()

    def create_deployment_map(self) -> tuple[list[DeploymentZone], list[HarvestZone]]:
        """Pre-compute deployment/harvest zones from track metadata."""
        deployment: list[DeploymentZone] = []
        harvest: list[HarvestZone] = []

        sectors = self.track_data.get("sectors", [])
        if sectors:
            for sector in sectors:
                start = float(sector.get("start", 0.0))
                end = float(sector.get("end", start + 300.0))
                segment_type = str(sector.get("type", "mixed"))
                if segment_type in {"straight", "exit"}:
                    deployment.append(
                        DeploymentZone(start=start, end=end, level="max_power")
                    )
                elif segment_type == "corner":
                    harvest.append(
                        HarvestZone(start=start, end=end, level="aggressive")
                    )
                else:
                    deployment.append(
                        DeploymentZone(start=start, end=end, level="balanced")
                    )
                    harvest.append(HarvestZone(start=start, end=end, level="balanced"))
        else:
            # Fallback generic map when detailed track segmentation is unavailable.
            deployment = [
                DeploymentZone(0.08, 0.22, "max_power"),
                DeploymentZone(0.38, 0.52, "max_power"),
                DeploymentZone(0.72, 0.9, "balanced"),
            ]
            harvest = [
                HarvestZone(0.22, 0.32, "aggressive"),
                HarvestZone(0.52, 0.64, "aggressive"),
                HarvestZone(0.9, 1.0, "balanced"),
            ]
        return deployment, harvest

    def _normalize_position(self, track_position: float) -> float:
        if track_position > 1.0:
            return float(
                (track_position % self.track_length) / max(self.track_length, 1.0)
            )
        return float(track_position % 1.0)

    def _in_zone(self, pos: float, start: float, end: float) -> bool:
        if start <= end:
            return start <= pos <= end
        return pos >= start or pos <= end

    def get_deployment_strategy(
        self,
        track_position: float,
        battery_soc: float,
        lap_type: str = "race",
    ) -> str:
        """Select deployment strategy at current track position."""
        pos = self._normalize_position(track_position)
        soc = float(np.clip(battery_soc, 0.0, 1.0))

        strategy = "balanced"
        for zone in self.deployment_map:
            if self._in_zone(pos, zone.start, zone.end):
                strategy = zone.level
                break

        if lap_type == "qualifying" and soc > 0.5 and strategy != "save_energy":
            strategy = "max_power"

        if soc < 0.2:
            strategy = "save_energy"
        elif soc < 0.35 and strategy == "max_power":
            strategy = "balanced"
        return strategy

    def get_harvest_strategy(self, track_position: float, battery_soc: float) -> str:
        """Select harvesting strategy based on position and SOC."""
        pos = self._normalize_position(track_position)
        soc = float(np.clip(battery_soc, 0.0, 1.0))

        strategy = "balanced"
        for zone in self.harvest_map:
            if self._in_zone(pos, zone.start, zone.end):
                strategy = zone.level
                break

        if soc > 0.9:
            return "conservative"
        if soc < 0.25:
            return "aggressive"
        return strategy

    def get_control_actions(
        self,
        track_position: float,
        battery_soc: float,
        lap_type: str = "race",
    ) -> dict[str, str]:
        """Convenience helper returning both deploy and harvest directives."""
        return {
            "deployment_strategy": self.get_deployment_strategy(
                track_position, battery_soc, lap_type
            ),
            "harvest_strategy": self.get_harvest_strategy(track_position, battery_soc),
        }
