#!/usr/bin/env python3
"""Common utilities for repository bootstrap scripts.

This module centralizes logging, command execution, report writing, and
cross-platform helpers used by Step 1.5 scripts.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CmdResult:
    """Structured result for shell command execution."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def utc_now() -> str:
    """Return ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def repo_root() -> Path:
    """Return repository root from this scripts module location."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    """Create directory tree if missing and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(
    name: str, log_dir: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure logger with console + file outputs.

    Logs are written under logs/bootstrap by default.
    """
    root = repo_root()
    out_dir = ensure_dir(log_dir or (root / "logs" / "bootstrap"))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(level)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(out_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def run_cmd(
    command: list[str],
    cwd: Path | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> CmdResult:
    """Run command and capture stdout/stderr.

    Args:
        command: Command tokens.
        cwd: Optional working directory.
        check: Raise RuntimeError on non-zero when True.
        env: Optional environment overrides.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )
    result = CmdResult(
        returncode=proc.returncode,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
        command=command,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def is_binary_file(path: Path, sample_size: int = 4096) -> bool:
    """Heuristic binary file detection via NUL bytes."""
    try:
        data = path.read_bytes()[:sample_size]
    except OSError:
        return False
    return b"\x00" in data


def human_size(num_bytes: int) -> str:
    """Convert byte count to readable IEC units."""
    size = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write JSON file with UTF-8 encoding."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write plain text UTF-8 file."""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def has_tty() -> bool:
    """Return true when running in interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def current_branch(root: Path | None = None) -> str:
    """Resolve current Git branch, fallback to main."""
    result = run_cmd(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root or repo_root()
    )
    if result.returncode != 0 or not result.stdout:
        return "main"
    return result.stdout


def safe_import(name: str):
    """Import module by name, return None on failure."""
    try:
        return __import__(name)
    except Exception:
        return None


def system_snapshot() -> dict[str, Any]:
    """Collect lightweight machine/environment metadata."""
    return {
        "timestamp": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
    }


def gitpython_repo(root: Path | None = None):
    """Return GitPython Repo object if available.

    Raises:
        RuntimeError when GitPython is unavailable.
    """
    try:
        from git import Repo  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "GitPython is required. Install with: pip install GitPython"
        ) from exc
    return Repo(str(root or repo_root()))


def copytree_filtered(
    source: Path, destination: Path, ignore: shutil.IgnorePattern | None = None
) -> None:
    """Copy directory tree with overwrite behavior."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=ignore)
