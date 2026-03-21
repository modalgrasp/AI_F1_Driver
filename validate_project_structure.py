#!/usr/bin/env python3
"""Validate repository structure, configs, imports, and Git setup."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

REQUIRED_DIRS = [
    ".github/workflows",
    "configs",
    "data",
    "docs",
    "environments",
    "experiments",
    "models",
    "scripts",
    "tests",
    "training",
    "utils",
    "vehicle_dynamics",
    "control",
    "planning",
    "perception",
]

REQUIRED_FILES = [
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "pytest.ini",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "LICENSE",
]


def check_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def check_python_syntax(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def check_git(root: Path) -> bool:
    if not (root / ".git").exists():
        return False
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return proc.returncode == 0


def main() -> int:
    root = Path(__file__).resolve().parent
    errors: list[str] = []

    for rel in REQUIRED_DIRS:
        if not (root / rel).exists():
            errors.append(f"Missing directory: {rel}")
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"Missing file: {rel}")

    for config in (root / "configs").glob("*.json"):
        if not check_json(config):
            errors.append(f"Invalid JSON: {config}")

    for py in root.rglob("*.py"):
        if ".venv" in py.parts or "f1_racing_env" in py.parts:
            continue
        if not check_python_syntax(py):
            errors.append(f"Syntax error: {py}")

    if not check_git(root):
        errors.append("Git repository not initialized or git status failed")

    report = {
        "ok": len(errors) == 0,
        "errors": errors,
    }
    out = root / "logs" / "project_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
