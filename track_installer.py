#!/usr/bin/env python3
"""Install Assetto Corsa track archives with backup and rollback support."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.config_manager import ConfigManager
from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)

try:
    import rarfile
except Exception:  # pragma: no cover
    rarfile = None  # type: ignore[assignment]


class TrackInstallError(RuntimeError):
    """Raised when track installation fails."""


class TrackInstaller:
    """Track installer with validation, backup, and rollback."""

    REQUIRED_MARKERS = [
        "models.ini",
        "data/surfaces.ini",
    ]

    def __init__(self, config_path: str | Path = "configs/config.json") -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        self.ac_root = self._resolve_ac_root()
        self.tracks_dir = self.ac_root / "content" / "tracks"
        self.backup_dir = Path("data/backups/tracks")
        self.install_log_dir = Path("logs")
        self.install_log_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_ac_root(self) -> Path:
        cfg = self.config.get("assetto_corsa", {})
        configured = Path(cfg.get("install_path", "")).expanduser() if cfg.get("install_path") else None
        if configured and configured.exists():
            return configured

        candidates = [
            Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa"),
            Path("C:/Program Files/Steam/steamapps/common/assettocorsa"),
            Path.home() / ".steam/steam/steamapps/common/assettocorsa",
            Path.home() / ".local/share/Steam/steamapps/common/assettocorsa",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise TrackInstallError("Assetto Corsa install path not found. Set assetto_corsa.install_path in config.")

    def install_archive(self, archive_path: Path, track_id: str | None = None) -> dict[str, Any]:
        """Install a track archive and return operation report."""
        archive_path = archive_path.resolve()
        if not archive_path.exists():
            raise TrackInstallError(f"Archive file not found: {archive_path}")

        self._validate_archive(archive_path)
        backup_path = self._create_backup()

        report: dict[str, Any] = {
            "archive": str(archive_path),
            "backup": str(backup_path),
            "installed_tracks": [],
            "status": "failed",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        with tempfile.TemporaryDirectory(prefix="yas_install_") as tmp_dir:
            temp_root = Path(tmp_dir)
            self._extract_archive(archive_path, temp_root)
            track_roots = self._detect_track_roots(temp_root)

            if track_id:
                filtered = [root for root in track_roots if self._track_id_matches(track_id, root.name)]
                if filtered:
                    track_roots = filtered
                else:
                    LOGGER.warning(
                        "No exact track-id match for '%s'. Available roots: %s. Proceeding with detected roots.",
                        track_id,
                        [root.name for root in track_roots],
                    )

            if not track_roots:
                raise TrackInstallError(
                    "No track roots detected in archive. Expected folder with models.ini and data/surfaces.ini."
                )

            try:
                self.tracks_dir.mkdir(parents=True, exist_ok=True)
                for source_root in track_roots:
                    target_root = self.tracks_dir / source_root.name
                    if target_root.exists():
                        shutil.rmtree(target_root)
                    shutil.copytree(source_root, target_root)
                    self._verify_installed_track(target_root)
                    report["installed_tracks"].append(source_root.name)

                self._update_content_index(report["installed_tracks"])
                report["status"] = "success"
                self._write_install_log(report)
                return report
            except Exception as exc:
                LOGGER.exception("Installation failed. Initiating rollback.")
                self.rollback_from_backup(backup_path)
                raise TrackInstallError(f"Installation failed and rollback executed: {exc}") from exc

    def rollback_latest(self) -> None:
        """Rollback from newest backup snapshot."""
        snapshots = sorted(self.backup_dir.glob("tracks_backup_*"), reverse=True)
        if not snapshots:
            raise TrackInstallError("No backup snapshots found for rollback.")
        self.rollback_from_backup(snapshots[0])

    def rollback_from_backup(self, backup_path: Path) -> None:
        """Restore tracks directory from backup."""
        if not backup_path.exists():
            raise TrackInstallError(f"Backup path not found: {backup_path}")
        if self.tracks_dir.exists():
            shutil.rmtree(self.tracks_dir)
        shutil.copytree(backup_path, self.tracks_dir)
        LOGGER.warning("Rollback complete from %s", backup_path)

    def _validate_archive(self, archive_path: Path) -> None:
        suffix = archive_path.suffix.lower()
        if suffix not in {".zip", ".rar"}:
            raise TrackInstallError("Only .zip and .rar archives are supported.")

        if suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise TrackInstallError(f"Archive integrity error at file: {bad}")
        elif suffix == ".rar":
            if rarfile is None:
                raise TrackInstallError("rarfile package not available. Install it to handle .rar archives.")
            with rarfile.RarFile(archive_path, "r") as rf:
                bad = rf.testrar()
                if bad is not None:
                    raise TrackInstallError(f"RAR integrity check failed: {bad}")

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        suffix = archive_path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.infolist()
                total = max(1, len(members))
                for idx, member in enumerate(members, start=1):
                    zf.extract(member, path=target_dir)
                    if idx % 50 == 0 or idx == total:
                        LOGGER.info("ZIP extraction progress: %d/%d", idx, total)
        else:
            if rarfile is None:
                raise TrackInstallError("rarfile package not available.")
            with rarfile.RarFile(archive_path, "r") as rf:
                members = rf.infolist()
                total = max(1, len(members))
                for idx, member in enumerate(members, start=1):
                    rf.extract(member, path=target_dir)
                    if idx % 50 == 0 or idx == total:
                        LOGGER.info("RAR extraction progress: %d/%d", idx, total)

    def _detect_track_roots(self, extraction_root: Path) -> list[Path]:
        candidates: list[Path] = []

        # Case A: classic single-layout track root containing models.ini + data/surfaces.ini.
        for models_path in extraction_root.rglob("models.ini"):
            candidate = models_path.parent
            if (candidate / "data" / "surfaces.ini").exists():
                candidates.append(candidate)

        # Case B: multi-layout root containing models_<layout>.ini and layout folders.
        for models_layout_path in extraction_root.rglob("models_*.ini"):
            root = models_layout_path.parent
            layout_dirs = self._detect_layout_dirs(root)
            if layout_dirs:
                candidates.append(root)

        # Deduplicate by resolved path.
        unique: dict[str, Path] = {}
        for root in candidates:
            try:
                unique[str(root.resolve())] = root
            except OSError:
                unique[str(root)] = root
        return list(unique.values())

    def _create_backup(self) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"tracks_backup_{timestamp}"
        if self.tracks_dir.exists():
            shutil.copytree(self.tracks_dir, backup_path)
        else:
            backup_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Tracks backup created at %s", backup_path)
        return backup_path

    def _verify_installed_track(self, track_root: Path) -> None:
        layout_dirs = self._detect_layout_dirs(track_root)
        if not layout_dirs:
            raise TrackInstallError(
                f"Track {track_root.name} has no valid layout directories with data/surfaces.ini"
            )

        map_found = False
        ai_found = False
        for layout in layout_dirs:
            if any((layout / name).exists() for name in ["map.png", "map.jpg", "ui/map.png", "ui/map.jpg"]):
                map_found = True
            if any((layout / name).exists() for name in ["ai/fast_lane.ai", "ai/fast_lane.csv"]):
                ai_found = True

        if not map_found:
            raise TrackInstallError(f"Track {track_root.name} has no layout map image file.")
        if not ai_found:
            raise TrackInstallError(f"Track {track_root.name} has no layout AI lane file.")

    def _detect_layout_dirs(self, track_root: Path) -> list[Path]:
        # Single-layout tracks place data directly in root.
        if (track_root / "data" / "surfaces.ini").exists():
            return [track_root]

        # Multi-layout tracks place data under layout subfolders (e.g., gp/data/surfaces.ini).
        layout_dirs: list[Path] = []
        for child in track_root.iterdir() if track_root.exists() else []:
            if child.is_dir() and (child / "data" / "surfaces.ini").exists():
                layout_dirs.append(child)
        return layout_dirs

    @staticmethod
    def _normalize_track_id(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _track_id_matches(self, requested_id: str, detected_name: str) -> bool:
        req = self._normalize_track_id(requested_id)
        det = self._normalize_track_id(detected_name)
        return req == det or req in det or det in req

    def _update_content_index(self, installed_tracks: list[str]) -> None:
        index_path = Path("data") / "installed_tracks_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "tracks": sorted(set(installed_tracks)),
            "ac_tracks_dir": str(self.tracks_dir),
        }
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Content index updated at %s", index_path)

    def _write_install_log(self, report: dict[str, Any]) -> None:
        log_path = self.install_log_dir / "track_install_report.json"
        log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install Assetto Corsa track archives safely.")
    parser.add_argument("--archive", type=Path, help="Path to .zip or .rar track archive")
    parser.add_argument("--track-id", type=str, default=None, help="Optional explicit track folder name")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"), help="Config path")
    parser.add_argument("--rollback-latest", action="store_true", help="Rollback using latest backup snapshot")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(Path("logs"), level="INFO", console=True)

    installer = TrackInstaller(config_path=args.config)
    if args.rollback_latest:
        installer.rollback_latest()
        LOGGER.info("Rollback complete.")
        return 0

    if args.archive is None:
        raise SystemExit("--archive is required unless --rollback-latest is used")

    report = installer.install_archive(archive_path=args.archive, track_id=args.track_id)
    LOGGER.info("Track installation finished: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
