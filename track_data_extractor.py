#!/usr/bin/env python3
"""Extract track geometry and metadata from Assetto Corsa track files."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils.config_manager import ConfigManager
from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    width: float
    speed_hint_kmh: float
    sector: int


class TrackDataExtractor:
    """Extract AI line and supporting geometry with graceful fallbacks."""

    def __init__(
        self,
        config_path: str | Path = "configs/config.json",
        track_id: str = "yas_marina",
    ) -> None:
        config = ConfigManager(config_path).load()
        self.track_id = track_id
        self.track_root = self._resolve_track_root(
            Path(config["assetto_corsa"]["install_path"]), track_id
        )
        self.layout_dirs = self._layout_dirs(self.track_root)
        self.primary_layout = self._select_primary_layout(self.layout_dirs)
        self.output_root = Path("data/tracks") / track_id / "extracted"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _resolve_track_root(self, ac_root: Path, track_id: str) -> Path:
        direct = ac_root / "content" / "tracks" / track_id
        if direct.exists():
            return direct

        tracks_dir = ac_root / "content" / "tracks"
        req = self._normalize_track_id(track_id)
        if tracks_dir.exists():
            for path in tracks_dir.iterdir():
                if not path.is_dir():
                    continue
                det = self._normalize_track_id(path.name)
                if req == det or req in det or det in req:
                    return path

        return direct

    @staticmethod
    def _normalize_track_id(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def _layout_dirs(track_root: Path) -> list[Path]:
        if (track_root / "data" / "surfaces.ini").exists():
            return [track_root]
        layouts: list[Path] = []
        for child in track_root.iterdir() if track_root.exists() else []:
            if child.is_dir() and (child / "data" / "surfaces.ini").exists():
                layouts.append(child)
        return layouts

    @staticmethod
    def _select_primary_layout(layouts: list[Path]) -> Path | None:
        if not layouts:
            return None
        for layout in layouts:
            if layout.name.lower() in {"gp", "grandprix", "grand_prix"}:
                return layout
        return layouts[0]

    def extract(self) -> dict[str, Any]:
        if not self.track_root.exists():
            raise FileNotFoundError(f"Track root not found: {self.track_root}")

        waypoints = self._extract_waypoints()
        surfaces = self._extract_surfaces()
        sectors = self._extract_sectors(waypoints)
        drs_zones = self._extract_drs_zones()
        pit = self._extract_pit_lane()
        turns = self._extract_turns(waypoints)
        boundaries = self._extract_boundaries(waypoints)
        elevation = self._extract_elevation(waypoints)

        payload = {
            "metadata": {
                "track_id": self.track_id,
                "track_root": str(self.track_root),
                "extracted_at": datetime.now(UTC).isoformat(),
                "coordinate_system": "Assetto Corsa right-handed, meters",
                "waypoint_count": len(waypoints),
            },
            "waypoints": [asdict(wp) for wp in waypoints],
            "surfaces": surfaces,
            "boundaries": boundaries,
            "drs_zones": drs_zones,
            "turns": turns,
            "sectors": sectors,
            "pit_lane": pit,
            "elevation_profile": elevation,
        }

        self._save_all_formats(payload, waypoints)
        return payload

    def _extract_waypoints(self) -> list[Waypoint]:
        search_roots = (
            ([self.primary_layout] if self.primary_layout else [])
            + self.layout_dirs
            + [self.track_root]
        )

        csv_lane = next(
            (
                root / "ai" / "fast_lane.csv"
                for root in search_roots
                if root and (root / "ai" / "fast_lane.csv").exists()
            ),
            None,
        )
        ai_lane = next(
            (
                root / "ai" / "fast_lane.ai"
                for root in search_roots
                if root and (root / "ai" / "fast_lane.ai").exists()
            ),
            None,
        )

        if csv_lane is not None and csv_lane.exists():
            return self._parse_fast_lane_csv(csv_lane)

        if ai_lane is not None and ai_lane.exists():
            # fast_lane.ai can be proprietary/binary depending on mod toolchain.
            # We intentionally provide a fallback parser strategy.
            parsed = self._parse_fast_lane_binary_fallback(ai_lane)
            if parsed and self._waypoints_plausible(parsed):
                return parsed
            if parsed:
                LOGGER.warning(
                    "Binary AI parse deemed implausible; using synthetic fallback."
                )

        LOGGER.warning(
            "No parseable AI lane found. Building synthetic centerline fallback."
        )
        return self._synthetic_waypoints()

    def _waypoints_plausible(self, waypoints: list[Waypoint]) -> bool:
        if len(waypoints) < 300 or len(waypoints) > 30000:
            return False
        xyz = np.array([[wp.x, wp.y, wp.z] for wp in waypoints], dtype=np.float64)
        d = np.sqrt(np.sum(np.diff(xyz[:, [0, 2]], axis=0) ** 2, axis=1))
        lap_len = float(np.sum(np.clip(d, 0.0, 1000.0)))
        return 2000.0 <= lap_len <= 12000.0

    def _parse_fast_lane_csv(self, path: Path) -> list[Waypoint]:
        waypoints: list[Waypoint] = []
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                waypoints.append(
                    Waypoint(
                        x=float(row.get("x", 0.0)),
                        y=float(row.get("y", 0.0)),
                        z=float(row.get("z", 0.0)),
                        width=float(row.get("width", 12.0)),
                        speed_hint_kmh=float(row.get("speed_hint_kmh", 220.0)),
                        sector=int(row.get("sector", 1)),
                    )
                )
        if waypoints:
            return waypoints
        return self._synthetic_waypoints()

    def _parse_fast_lane_binary_fallback(self, path: Path) -> list[Waypoint]:
        """Best-effort parser for unknown binary AI format.

        This method attempts to interpret data as repeated float triplets and filters
        unrealistic points. For production, replace with a dedicated AC AI parser.
        """
        blob = path.read_bytes()
        if len(blob) < 48:
            return []

        floats = np.frombuffer(blob, dtype=np.float32)
        if floats.size < 30:
            return []

        triples = floats[: (floats.size // 3) * 3].reshape(-1, 3)
        triples = triples[np.isfinite(triples).all(axis=1)]

        # Filter unreasonable coordinates (mod-dependent but avoids garbage vectors).
        mask = np.logical_and(np.abs(triples) < 1e6, True).all(axis=1)
        triples = triples[mask]
        if triples.shape[0] < 100:
            return []

        waypoints = [
            Waypoint(
                x=float(row[0]),
                y=float(row[1]),
                z=float(row[2]),
                width=12.0,
                speed_hint_kmh=220.0,
                sector=1,
            )
            for row in triples
        ]
        LOGGER.warning(
            "Binary fast_lane.ai parsed with fallback heuristic. Verify geometry manually."
        )
        return waypoints

    def _synthetic_waypoints(self, count: int = 1200) -> list[Waypoint]:
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        a = 900.0
        b = 520.0
        x = a * np.cos(theta)
        z = b * np.sin(theta)
        y = 2.0 * np.sin(theta * 3.0)
        sectors = np.digitize(np.linspace(0.0, 1.0, count), [1 / 3, 2 / 3]) + 1
        return [
            Waypoint(float(xi), float(yi), float(zi), 12.0, 220.0, int(sec))
            for xi, yi, zi, sec in zip(x, y, z, sectors)
        ]

    def _extract_surfaces(self) -> list[dict[str, Any]]:
        search_roots = (
            ([self.primary_layout] if self.primary_layout else [])
            + self.layout_dirs
            + [self.track_root]
        )
        path = next(
            (
                root / "data" / "surfaces.ini"
                for root in search_roots
                if root and (root / "data" / "surfaces.ini").exists()
            ),
            None,
        )
        if path is None or not path.exists():
            return []

        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        surfaces: list[dict[str, Any]] = []
        for section in parser.sections():
            if not section.upper().startswith("SURFACE"):
                continue
            surface = {
                "id": section,
                "key": parser.get(section, "KEY", fallback="UNKNOWN"),
                "friction": parser.getfloat(section, "FRICTION", fallback=0.98),
                "damping": parser.getfloat(section, "DAMPING", fallback=0.0),
                "is_valid_track": parser.getboolean(
                    section, "IS_VALID_TRACK", fallback=True
                ),
            }
            surfaces.append(surface)
        return surfaces

    def _extract_boundaries(
        self, waypoints: list[Waypoint]
    ) -> dict[str, list[list[float]]]:
        xyz = np.array([[wp.x, wp.y, wp.z] for wp in waypoints], dtype=np.float64)
        widths = np.array([wp.width for wp in waypoints], dtype=np.float64)

        diff = np.roll(xyz[:, [0, 2]], -1, axis=0) - xyz[:, [0, 2]]
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        tangents = diff / norms
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        left_xz = xyz[:, [0, 2]] + normals * (widths[:, None] / 2.0)
        right_xz = xyz[:, [0, 2]] - normals * (widths[:, None] / 2.0)

        left = np.column_stack([left_xz[:, 0], xyz[:, 1], left_xz[:, 1]])
        right = np.column_stack([right_xz[:, 0], xyz[:, 1], right_xz[:, 1]])

        return {
            "centerline": xyz.tolist(),
            "left": left.tolist(),
            "right": right.tolist(),
        }

    def _extract_drs_zones(self) -> list[dict[str, float]]:
        search_roots = (
            ([self.primary_layout] if self.primary_layout else [])
            + self.layout_dirs
            + [self.track_root]
        )
        path = next(
            (
                root / "data" / "drs_zones.ini"
                for root in search_roots
                if root and (root / "data" / "drs_zones.ini").exists()
            ),
            None,
        )
        if path is None or not path.exists():
            # Yas Marina default placeholders.
            return [
                {"zone": 1, "start_m": 1700.0, "end_m": 2500.0, "activation_m": 1600.0},
                {"zone": 2, "start_m": 3200.0, "end_m": 3900.0, "activation_m": 3100.0},
            ]

        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        zones: list[dict[str, float]] = []
        for section in parser.sections():
            if "DRS" in section.upper():
                zones.append(
                    {
                        "zone": float(
                            parser.get(section, "ID", fallback=len(zones) + 1)
                        ),
                        "start_m": parser.getfloat(section, "START", fallback=0.0),
                        "end_m": parser.getfloat(section, "END", fallback=0.0),
                        "activation_m": parser.getfloat(
                            section, "ACTIVATION", fallback=0.0
                        ),
                    }
                )
        return zones

    def _extract_turns(self, waypoints: list[Waypoint]) -> list[dict[str, Any]]:
        xyz = np.array([[wp.x, wp.y, wp.z] for wp in waypoints], dtype=np.float64)
        curvature = self._curvature(xyz[:, [0, 2]])
        peaks = np.where(curvature > np.percentile(curvature, 92))[0]

        turns: list[dict[str, Any]] = []
        if len(peaks) == 0:
            return turns

        grouped = np.split(peaks, np.where(np.diff(peaks) > 20)[0] + 1)
        for idx, group in enumerate(grouped[:20], start=1):
            apex_idx = int(group[np.argmax(curvature[group])])
            radius = float(max(1e-6, 1.0 / curvature[apex_idx]))
            turns.append(
                {
                    "turn": idx,
                    "apex_index": apex_idx,
                    "apex_xyz": xyz[apex_idx].tolist(),
                    "radius_m": radius,
                    "type": self._classify_turn(radius),
                    "banking_deg": 0.0,
                }
            )
        return turns

    def _extract_sectors(self, waypoints: list[Waypoint]) -> list[dict[str, Any]]:
        sectors = sorted(set(wp.sector for wp in waypoints))
        return [{"sector": sec, "name": f"Sector {sec}"} for sec in sectors]

    def _extract_pit_lane(self) -> dict[str, Any]:
        pit_data = {
            "entry_xyz": None,
            "exit_xyz": None,
            "length_m": None,
            "speed_limit_kmh_practice_qualifying": 80,
            "speed_limit_kmh_race": 60,
        }
        search_roots = (
            ([self.primary_layout] if self.primary_layout else [])
            + self.layout_dirs
            + [self.track_root]
        )
        path = next(
            (
                root / "ai" / "pit_lane.csv"
                for root in search_roots
                if root and (root / "ai" / "pit_lane.csv").exists()
            ),
            None,
        )
        if path is not None and path.exists():
            rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
            if rows:
                first = rows[0]
                last = rows[-1]
                entry = [
                    float(first.get("x", 0.0)),
                    float(first.get("y", 0.0)),
                    float(first.get("z", 0.0)),
                ]
                exit_ = [
                    float(last.get("x", 0.0)),
                    float(last.get("y", 0.0)),
                    float(last.get("z", 0.0)),
                ]
                pit_data["entry_xyz"] = entry
                pit_data["exit_xyz"] = exit_
        else:
            # If CSV unavailable, indicate pit lane presence when AI binary exists.
            ai_pit = next(
                (
                    root / "ai" / "pit_lane.ai"
                    for root in search_roots
                    if root and (root / "ai" / "pit_lane.ai").exists()
                ),
                None,
            )
            if ai_pit is not None:
                pit_data["entry_xyz"] = [0.0, 0.0, 0.0]
                pit_data["exit_xyz"] = [0.0, 0.0, 0.0]
        return pit_data

    def _extract_elevation(self, waypoints: list[Waypoint]) -> dict[str, Any]:
        y = np.array([wp.y for wp in waypoints], dtype=np.float64)
        xyz = np.array([[wp.x, wp.y, wp.z] for wp in waypoints], dtype=np.float64)
        d = np.sqrt(np.sum(np.diff(xyz[:, [0, 2]], axis=0) ** 2, axis=1))
        s = np.insert(np.cumsum(d), 0, 0.0)
        gradient = np.gradient(y, s, edge_order=1)
        return {
            "distance_m": s.tolist(),
            "height_m": y.tolist(),
            "gradient": np.nan_to_num(gradient).tolist(),
        }

    @staticmethod
    def _classify_turn(radius_m: float) -> str:
        if radius_m < 50.0:
            return "slow"
        if radius_m < 150.0:
            return "medium"
        return "fast"

    @staticmethod
    def _curvature(points_xz: np.ndarray) -> np.ndarray:
        x = points_xz[:, 0]
        z = points_xz[:, 1]
        dx = np.gradient(x)
        dz = np.gradient(z)
        ddx = np.gradient(dx)
        ddz = np.gradient(dz)
        num = np.abs(dx * ddz - dz * ddx)
        den = np.power(dx * dx + dz * dz, 1.5)
        den[den == 0.0] = np.inf
        return num / den

    def _save_all_formats(
        self, payload: dict[str, Any], waypoints: list[Waypoint]
    ) -> None:
        json_path = self.output_root / f"{self.track_id}_track_data.json"
        csv_path = self.output_root / f"{self.track_id}_waypoints.csv"
        npz_path = self.output_root / f"{self.track_id}_track_arrays.npz"
        pkl_path = self.output_root / f"{self.track_id}_track_data.pkl"

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["x", "y", "z", "width", "speed_hint_kmh", "sector"],
            )
            writer.writeheader()
            for wp in waypoints:
                writer.writerow(asdict(wp))

        xyz = np.array([[w.x, w.y, w.z] for w in waypoints], dtype=np.float32)
        widths = np.array([w.width for w in waypoints], dtype=np.float32)
        speeds = np.array([w.speed_hint_kmh for w in waypoints], dtype=np.float32)
        np.savez_compressed(npz_path, xyz=xyz, width=widths, speed_hint_kmh=speeds)

        with pkl_path.open("wb") as fh:
            pickle.dump(payload, fh)

        LOGGER.info("Extraction outputs saved to %s", self.output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract track data for RL processing."
    )
    parser.add_argument("--track-id", type=str, default="yas_marina")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(Path("logs"), level="INFO", console=True)
    extractor = TrackDataExtractor(config_path=args.config, track_id=args.track_id)
    payload = extractor.extract()
    LOGGER.info("Extracted %d waypoints", payload["metadata"]["waypoint_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
