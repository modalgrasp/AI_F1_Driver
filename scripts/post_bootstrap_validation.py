#!/usr/bin/env python3
"""Final post-bootstrap verification and Phase 1 completion marker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import (
    repo_root,
    run_cmd,
    setup_logger,
    utc_now,
    write_json,
    write_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-bootstrap validation checks")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/bootstrap"))
    parser.add_argument("--create-marker", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("post_bootstrap_validation")
    root = repo_root()

    checks = {
        "git_repo": (root / ".git").exists(),
        "has_commit": run_cmd(["git", "rev-parse", "HEAD"], cwd=root).returncode == 0,
        "remote_configured": bool(run_cmd(["git", "remote"], cwd=root).stdout.strip()),
        "hooks_installed": all(
            (root / ".git" / "hooks" / name).exists()
            for name in ["pre-commit", "commit-msg", "pre-push"]
        ),
        "tests_report_exists": (
            root / "logs" / "bootstrap" / "phase1_test_report.json"
        ).exists(),
        "working_tree_clean": run_cmd(
            ["git", "status", "--porcelain"], cwd=root
        ).stdout.strip()
        == "",
    }

    ok = all(checks.values())
    report = {
        "timestamp": utc_now(),
        "ok": ok,
        "checks": checks,
        "final_status": "PHASE1_READY" if ok else "PHASE1_NOT_READY",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "post_bootstrap_validation.json", report)

    marker_path = root / "PHASE1_COMPLETE.marker"
    if args.create_marker and ok:
        write_text(marker_path, "Phase 1 complete and validated. Ready for Phase 2.\n")
        logger.info("Created marker file: %s", marker_path)

    status_path = root / "docs" / "reports" / "phase2_preparation_checklist.md"
    lines = [
        "# Phase 2 Preparation Checklist",
        "",
        f"- Bootstrap status: {'OK' if ok else 'NOT OK'}",
        "- [ ] Finalize vehicle dynamics model interfaces",
        "- [ ] Define tire model equations (Pacejka)",
        "- [ ] Add aerodynamics force model",
        "- [ ] Integrate powertrain energy constraints",
        "- [ ] Add dynamics-aware reward shaping",
    ]
    write_text(status_path, "\n".join(lines) + "\n")

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
