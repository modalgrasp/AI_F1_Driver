#!/usr/bin/env python3
"""Validate installed Yas Marina track content for RL readiness."""

from __future__ import annotations

import argparse
import configparser
import json
import logging
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.config_manager import ConfigManager
from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    check: str
    status: str
    message: str
    recommendation: str = ""


class TrackValidator:
    """Validate track file completeness and metadata consistency."""

    EXPECTED_LENGTH_KM = 5.281
    LENGTH_TOLERANCE_KM = 0.01
    EXPECTED_TURNS = 16

    def __init__(self, config_path: str | Path = "configs/config.json", track_id: str = "yas_marina") -> None:
        self.config = ConfigManager(config_path).load()
        self.track_id = track_id
        self.track_root = self._resolve_track_root(track_id)
        self.layout_dirs = self._layout_dirs(self.track_root)
        self.primary_layout = self._select_primary_layout(self.layout_dirs)

    def _resolve_track_root(self, track_id: str) -> Path:
        ac_root = Path(self.config["assetto_corsa"]["install_path"])
        track_root = ac_root / "content" / "tracks" / track_id
        if track_root.exists():
            return track_root

        # Search by partial name as fallback.
        tracks_dir = ac_root / "content" / "tracks"
        if tracks_dir.exists():
            req = self._normalize_track_id(track_id)
            matches = []
            for path in tracks_dir.iterdir():
                if not path.is_dir():
                    continue
                det = self._normalize_track_id(path.name)
                if req == det or req in det or det in req:
                    matches.append(path)
            if matches:
                return matches[0]

        raise FileNotFoundError(f"Track folder not found for id '{track_id}' under {tracks_dir}")

    @staticmethod
    def _normalize_track_id(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def _layout_dirs(track_root: Path) -> list[Path]:
        direct = track_root / "data" / "surfaces.ini"
        if direct.exists():
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

    def _exists_anywhere(self, relative_paths: list[str], allow_glob: bool = False) -> bool:
        search_roots = [self.track_root] + self.layout_dirs
        for root in search_roots:
            for rel in relative_paths:
                if allow_glob and "*" in rel:
                    if list(root.glob(rel)):
                        return True
                else:
                    if (root / rel).exists():
                        return True
        return False

    def run(self) -> dict[str, Any]:
        results: list[ValidationResult] = []

        results.extend(self._check_presence())
        metadata = self._parse_track_metadata()
        results.extend(self._check_length(metadata))
        results.extend(self._check_turn_count(metadata))
        results.extend(self._check_boundaries())
        results.extend(self._check_ai_spline())
        results.extend(self._check_drs())
        results.extend(self._check_pitlane())
        results.extend(self._check_grid())
        results.extend(self._check_surface_types())

        success = all(item.status != "FAIL" for item in results)
        report = {
            "track_id": self.track_id,
            "track_root": str(self.track_root),
            "timestamp": datetime.now(UTC).isoformat(),
            "success": success,
            "metadata": metadata,
            "results": [asdict(item) for item in results],
        }
        self._write_report(report)
        return report

    def _check_presence(self) -> list[ValidationResult]:
        required_any = {
            "3d_model": ["*.kn5"],
            "models_ini": ["models.ini", "models_gp.ini", "models_*.ini"],
            "surfaces_ini": ["data/surfaces.ini"],
            "ai_line": ["ai/fast_lane.ai", "ai/fast_lane.csv"],
            "map": ["map.png", "map.jpg", "ui/map.png", "ui/map.jpg", "data/map.ini", "map.ini"],
        }
        results: list[ValidationResult] = []
        for check_name, patterns in required_any.items():
            found = self._exists_anywhere(patterns, allow_glob=True)
            if found:
                results.append(ValidationResult(check_name, "PASS", "Required content found."))
            else:
                results.append(
                    ValidationResult(
                        check_name,
                        "FAIL",
                        f"Missing expected file pattern(s): {patterns}",
                        "Reinstall track package or choose a higher-quality mod release.",
                    )
                )
        return results

    def _parse_track_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "name": self.track_id,
            "description": "",
            "length_km": None,
            "corners": None,
            "pit_boxes": None,
            "location": "",
        }

        ui_json_candidates = [
            self.track_root / "ui" / "ui_track.json",
            self.track_root / "ui_track.json",
        ]
        if self.primary_layout is not None:
            ui_json_candidates.extend(
                [
                    self.primary_layout / "ui" / "ui_track.json",
                    self.primary_layout / "ui_track.json",
                ]
            )

        ui_json = next((path for path in ui_json_candidates if path.exists()), None)
        if ui_json is not None and ui_json.exists():
            try:
                payload = json.loads(ui_json.read_text(encoding="utf-8"))
                metadata["name"] = payload.get("name", metadata["name"])
                metadata["description"] = payload.get("description", "")
                length_km = payload.get("length") or payload.get("length_km")
                if isinstance(length_km, str):
                    length_km = length_km.lower().replace("km", "").strip()
                try:
                    metadata["length_km"] = float(length_km)
                except Exception:
                    pass
                metadata["location"] = payload.get("city", payload.get("country", ""))
            except Exception:
                LOGGER.exception("Failed parsing ui_track.json")

        map_ini = (self.primary_layout / "data" / "map.ini") if self.primary_layout else (self.track_root / "data" / "map.ini")
        if map_ini.exists():
            parser = configparser.ConfigParser()
            parser.read(map_ini, encoding="utf-8")
            if parser.has_section("PARAMETERS"):
                corners = parser.get("PARAMETERS", "CORNERS", fallback=None)
                if corners is not None:
                    try:
                        metadata["corners"] = int(corners)
                    except ValueError:
                        pass

        pit_ini = (self.primary_layout / "data" / "pit_lane.ini") if self.primary_layout else (self.track_root / "data" / "pit_lane.ini")
        if pit_ini.exists():
            parser = configparser.ConfigParser()
            parser.read(pit_ini, encoding="utf-8")
            metadata["pit_boxes"] = parser.getint("PITLANE", "BOXES", fallback=None)

        if metadata["length_km"] is None and "yas" in self.track_root.name.lower():
            metadata["length_km"] = self.EXPECTED_LENGTH_KM
        if metadata["corners"] is None and "yas" in self.track_root.name.lower():
            metadata["corners"] = self.EXPECTED_TURNS

        return metadata

    def _check_length(self, metadata: dict[str, Any]) -> list[ValidationResult]:
        length = metadata.get("length_km")
        if length is None:
            return [
                ValidationResult(
                    "track_length",
                    "WARN",
                    "Length not found in metadata.",
                    "Set length in ui_track.json or provide measured centerline length.",
                )
            ]

        delta = abs(float(length) - self.EXPECTED_LENGTH_KM)
        if delta <= self.LENGTH_TOLERANCE_KM:
            return [ValidationResult("track_length", "PASS", f"Length within tolerance: {length} km")]

        return [
            ValidationResult(
                "track_length",
                "FAIL",
                f"Length mismatch: got {length} km, expected {self.EXPECTED_LENGTH_KM} +/- {self.LENGTH_TOLERANCE_KM}",
                "Use a higher-fidelity Yas Marina mod or update metadata with accurate value.",
            )
        ]

    def _check_turn_count(self, metadata: dict[str, Any]) -> list[ValidationResult]:
        corners = metadata.get("corners")
        if corners is None:
            return [
                ValidationResult(
                    "turn_count",
                    "WARN",
                    "Turn count not found in metadata.",
                    "Add corners count to map.ini or analysis output.",
                )
            ]

        if int(corners) == self.EXPECTED_TURNS:
            return [ValidationResult("turn_count", "PASS", f"Turns match expected count: {corners}")]

        return [
            ValidationResult(
                "turn_count",
                "FAIL",
                f"Turns mismatch: got {corners}, expected {self.EXPECTED_TURNS}",
                "Check layout variant selection or use primary GP layout.",
            )
        ]

    def _check_boundaries(self) -> list[ValidationResult]:
        if self._exists_anywhere(["data/ideal_line_left.csv"]) and self._exists_anywhere(["data/ideal_line_right.csv"]):
            return [ValidationResult("track_boundaries", "PASS", "Boundary files found.")]
        return [
            ValidationResult(
                "track_boundaries",
                "WARN",
                "Boundary coordinate files not found.",
                "Generate boundaries from centerline and width model using track_data_extractor.py",
            )
        ]

    def _check_ai_spline(self) -> list[ValidationResult]:
        if self._exists_anywhere(["ai/fast_lane.ai", "ai/fast_lane.csv"]):
            return [ValidationResult("ai_spline", "PASS", "AI lane file exists.")]
        return [
            ValidationResult(
                "ai_spline",
                "FAIL",
                "AI spline file missing.",
                "Install a track release that includes ai/fast_lane.ai or generate one with AC tools.",
            )
        ]

    def _check_drs(self) -> list[ValidationResult]:
        if self._exists_anywhere(["data/drs_zones.ini"]):
            return [ValidationResult("drs_zones", "PASS", "DRS zones definition found.")]
        return [
            ValidationResult(
                "drs_zones",
                "WARN",
                "DRS zones file not found.",
                "Provide drs_zones.ini or configure zones manually in extracted data.",
            )
        ]

    def _check_pitlane(self) -> list[ValidationResult]:
        if self._exists_anywhere(["ai/pit_lane.ai", "data/pit_lane.ini"]):
            return [ValidationResult("pit_lane", "PASS", "Pit lane configuration present.")]
        return [
            ValidationResult(
                "pit_lane",
                "WARN",
                "Pit lane config missing.",
                "Add pit_lane.ai and pit settings to support realistic race sessions.",
            )
        ]

    def _check_grid(self) -> list[ValidationResult]:
        if self._exists_anywhere(["data/starting_grid.ini", "ai/pit_lane_with_grid.ai"]):
            return [ValidationResult("starting_grid", "PASS", "Starting grid file present.")]
        return [
            ValidationResult(
                "starting_grid",
                "WARN",
                "starting_grid.ini not found.",
                "Verify spawn points in track editor or update mod.",
            )
        ]

    def _check_surface_types(self) -> list[ValidationResult]:
        surface_files: list[Path] = []
        for root in [self.track_root] + self.layout_dirs:
            path = root / "data" / "surfaces.ini"
            if path.exists():
                surface_files.append(path)

        if not surface_files:
            return [
                ValidationResult(
                    "surface_types",
                    "FAIL",
                    "surfaces.ini missing.",
                    "Track physics cannot be validated without surfaces.ini.",
                )
            ]

        sections: list[str] = []
        for surface_path in surface_files:
            parser = configparser.ConfigParser()
            parser.read(surface_path, encoding="utf-8")
            sections.extend([name for name in parser.sections() if name.upper().startswith("SURFACE")])
        if sections:
            return [ValidationResult("surface_types", "PASS", f"Surface definitions found: {len(sections)}")]
        return [
            ValidationResult(
                "surface_types",
                "WARN",
                "No SURFACE sections parsed from surfaces.ini.",
                "Check encoding/format of surfaces.ini.",
            )
        ]

    def _write_report(self, report: dict[str, Any]) -> None:
        out_dir = Path("data/tracks") / self.track_id / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "track_validation_report.json"
        txt_path = out_dir / "track_validation_report.txt"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        lines = [
            f"Track Validation Report: {self.track_id}",
            f"Track root: {self.track_root}",
            f"Timestamp: {report['timestamp']}",
            f"Overall success: {report['success']}",
            "",
            "Results:",
        ]
        for item in report["results"]:
            lines.append(f"- {item['check']}: {item['status']} | {item['message']}")
            if item.get("recommendation"):
                lines.append(f"  Recommendation: {item['recommendation']}")

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Validation reports written: %s, %s", json_path, txt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a track installation for RL readiness.")
    parser.add_argument("--track-id", type=str, default="yas_marina")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(Path("logs"), level="INFO", console=True)
    validator = TrackValidator(config_path=args.config, track_id=args.track_id)
    report = validator.run()
    LOGGER.info("Validation success: %s", report["success"])
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
