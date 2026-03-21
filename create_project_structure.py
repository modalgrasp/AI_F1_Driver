#!/usr/bin/env python3
"""Create and verify the expected F1 project directory structure."""

from __future__ import annotations

import os
from pathlib import Path

STRUCTURE = {
    "environments": ["__init__.py", "f1_racing_env.py", "assetto_corsa_connector.py"],
    "utils": [
        "__init__.py",
        "shared_memory_reader.py",
        "config_manager.py",
        "logger_config.py",
    ],
    "tests": ["__init__.py", "test_environment.py"],
    "logs": [".gitkeep"],
    "data": [".gitkeep"],
    "models": [".gitkeep"],
    "configs": ["config.json"],
    "docs": ["setup_guide.md"],
}

ROOT_FILES = [
    "requirements.txt",
    "setup.bat",
    "README.md",
    ".gitignore",
    "validate_setup.py",
    "create_project_structure.py",
]


def ensure_file(path: Path) -> None:
    """Create file if missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def set_exec_if_script(path: Path) -> None:
    """Apply executable permissions on non-Windows for scripts."""
    if os.name != "nt" and path.suffix in {".py", ".sh"}:
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)


def create_structure(root: Path) -> None:
    """Create project tree and placeholder files."""
    for folder, files in STRUCTURE.items():
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for filename in files:
            file_path = folder_path / filename
            ensure_file(file_path)
            set_exec_if_script(file_path)

    for filename in ROOT_FILES:
        file_path = root / filename
        ensure_file(file_path)
        set_exec_if_script(file_path)


def main() -> int:
    """Entrypoint."""
    root = Path(__file__).resolve().parent
    create_structure(root)
    print(f"Project structure verified at: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
