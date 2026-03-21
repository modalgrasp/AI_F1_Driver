#!/usr/bin/env python3
"""Backup and restore utilities for repository and configuration safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import repo_root, setup_logger, utc_now, write_json


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def backup_full(root: Path, destination: Path, include_artifacts: bool) -> Path:
    base_name = destination / archive_name("full_repo_backup")
    ignore = None
    if not include_artifacts:
        ignore = shutil.ignore_patterns("logs", "f1_racing_env", ".venv", "__pycache__", "*.pt", "*.pth")
    archive = shutil.make_archive(str(base_name), "zip", root_dir=str(root), base_dir=".", logger=None)
    return Path(archive)


def backup_config(root: Path, destination: Path) -> Path:
    temp = destination / archive_name("config_backup")
    temp.mkdir(parents=True, exist_ok=True)
    for rel in ["configs", ".gitignore", ".gitattributes", "requirements.txt", "requirements-dev.txt"]:
        src = root / rel
        if not src.exists():
            continue
        dst = temp / rel
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    archive = shutil.make_archive(str(temp), "zip", root_dir=str(temp), base_dir=".")
    shutil.rmtree(temp, ignore_errors=True)
    return Path(archive)


def backup_data(root: Path, destination: Path) -> Path:
    temp = destination / archive_name("data_backup")
    temp.mkdir(parents=True, exist_ok=True)
    candidates = [
        "data/tracks",
        "experiments",
        "models",
    ]
    for rel in candidates:
        src = root / rel
        if not src.exists():
            continue
        dst = temp / rel
        shutil.copytree(src, dst, dirs_exist_ok=True)
    archive = shutil.make_archive(str(temp), "zip", root_dir=str(temp), base_dir=".")
    shutil.rmtree(temp, ignore_errors=True)
    return Path(archive)


def restore(archive_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(output_dir))
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup and restore repository assets")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create backup archive")
    create.add_argument("--type", choices=["full", "config", "data"], default="full")
    create.add_argument("--destination", type=Path, default=Path("backups"))
    create.add_argument("--include-artifacts", action="store_true")

    restore_cmd = sub.add_parser("restore", help="Restore from backup archive")
    restore_cmd.add_argument("--archive", type=Path, required=True)
    restore_cmd.add_argument("--output-dir", type=Path, default=Path("restored_backup"))

    args = parser.parse_args()

    logger = setup_logger("backup_repository")
    root = repo_root()

    if args.command == "create":
        args.destination.mkdir(parents=True, exist_ok=True)
        if args.type == "full":
            archive = backup_full(root, args.destination, args.include_artifacts)
        elif args.type == "config":
            archive = backup_config(root, args.destination)
        else:
            archive = backup_data(root, args.destination)

        report = {
            "timestamp": utc_now(),
            "type": args.type,
            "archive": str(archive),
            "sha256": checksum(archive),
            "size_bytes": archive.stat().st_size,
            "schedule_recommendation": {
                "daily": "config",
                "weekly": "full",
                "retention": "keep last 14 backups",
            },
        }
        out = root / "logs" / "bootstrap" / "backup_report.json"
        write_json(out, report)
        logger.info("Backup archive created: %s", archive)
        print(json.dumps(report, indent=2))
        return 0

    restored = restore(args.archive, args.output_dir)
    report = {
        "timestamp": utc_now(),
        "archive": str(args.archive),
        "restored_to": str(restored),
    }
    out = root / "logs" / "bootstrap" / "restore_report.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
