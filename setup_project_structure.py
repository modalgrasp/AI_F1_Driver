#!/usr/bin/env python3
"""Generate and verify full project structure for collaborative development."""

from __future__ import annotations

from pathlib import Path

DIRS = [
    ".github/workflows",
    ".github/ISSUE_TEMPLATE",
    "configs",
    "data/raw",
    "data/processed",
    "data/track_maps",
    "data/telemetry",
    "docs/setup",
    "docs/architecture",
    "docs/api",
    "docs/tutorials",
    "docs/research_notes",
    "environments/wrappers",
    "experiments/experiment_001/checkpoints",
    "experiments/experiment_001/logs",
    "experiments/experiment_001/results",
    "experiments/experiment_template/checkpoints",
    "experiments/experiment_template/logs",
    "experiments/experiment_template/results",
    "models",
    "scripts",
    "tests",
    "training/algorithms",
    "utils",
    "vehicle_dynamics",
    "control",
    "planning",
    "perception",
    "notebooks",
]

INIT_FILES = [
    "configs/__init__.py",
    "environments/wrappers/__init__.py",
    "models/__init__.py",
    "training/__init__.py",
    "training/algorithms/__init__.py",
    "vehicle_dynamics/__init__.py",
    "control/__init__.py",
    "planning/__init__.py",
    "perception/__init__.py",
]

GITKEEP_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/track_maps/.gitkeep",
    "data/telemetry/.gitkeep",
    "experiments/experiment_001/checkpoints/.gitkeep",
    "experiments/experiment_001/logs/.gitkeep",
    "experiments/experiment_001/results/.gitkeep",
    "experiments/experiment_template/checkpoints/.gitkeep",
    "experiments/experiment_template/logs/.gitkeep",
    "experiments/experiment_template/results/.gitkeep",
]


def ensure_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parent
    for rel in DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for rel in INIT_FILES:
        ensure_file(root / rel, '"""Package init."""\n')
    for rel in GITKEEP_FILES:
        ensure_file(root / rel)

    print("Project structure setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
