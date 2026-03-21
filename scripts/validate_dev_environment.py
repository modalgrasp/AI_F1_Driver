#!/usr/bin/env python3
"""Validate local developer environment readiness for contributors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import repo_root, run_cmd, setup_logger, utc_now, write_json, write_text


TOOLS = ["git", "python", "pip", "nvidia-smi", "pre-commit", "pytest"]


def check_tool(name: str) -> dict[str, Any]:
    found = shutil.which(name)
    return {"tool": name, "available": found is not None, "path": found or ""}


def git_config(root: Path) -> dict[str, Any]:
    keys = ["user.name", "user.email", "core.editor", "merge.tool"]
    out = {}
    for key in keys:
        result = run_cmd(["git", "config", "--get", key], cwd=root)
        out[key] = result.stdout if result.returncode == 0 else ""
    return out


def ide_state(root: Path) -> dict[str, Any]:
    return {
        "vscode": (root / ".vscode").exists(),
        "pycharm": (root / ".idea").exists(),
    }


def python_env_state() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "path_head": os.environ.get("PATH", "").split(os.pathsep)[:5],
    }


def write_onboarding(root: Path) -> Path:
    path = root / "docs" / "reports" / "developer_onboarding_checklist.md"
    lines = [
        "# Developer Onboarding Checklist",
        "",
        "1. Clone repository",
        "2. Create and activate Python virtual environment",
        "3. Install runtime dependencies: `pip install -r requirements.txt`",
        "4. Install development dependencies: `pip install -r requirements-dev.txt`",
        "5. Install hooks: `pre-commit install --install-hooks`",
        "6. Run tests: `python scripts/run_phase1_tests.py --subset repo`",
        "7. Read contribution guidelines in CONTRIBUTING.md",
    ]
    write_text(path, "\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate developer workstation setup")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/bootstrap"))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("validate_dev_environment")
    root = repo_root()

    report = {
        "timestamp": utc_now(),
        "tools": [check_tool(name) for name in TOOLS],
        "git": git_config(root),
        "ide": ide_state(root),
        "python": python_env_state(),
        "workspace_clean": run_cmd(["git", "status", "--porcelain"], cwd=root).stdout.strip() == "",
    }
    guide = write_onboarding(root)
    report["onboarding_guide"] = str(guide)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dev_environment_report.json", report)
    logger.info("Developer environment report generated.")
    print(json.dumps(report, indent=2))

    missing_tools = [item["tool"] for item in report["tools"] if not item["available"]]
    return 1 if missing_tools else 0


if __name__ == "__main__":
    raise SystemExit(main())
