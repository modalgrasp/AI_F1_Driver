#!/usr/bin/env python3
"""Install and validate pre-commit hooks for repository quality gates."""

from __future__ import annotations

import argparse
import json
import re
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

HOOK_NAMES = [
    "black",
    "isort",
    "flake8",
    "mypy",
    "trailing-whitespace",
    "end-of-file-fixer",
    "check-yaml",
    "check-json",
    "check-added-large-files",
]


AUTO_FIX_HOOKS = {"black", "isort", "trailing-whitespace", "end-of-file-fixer"}


def ensure_precommit_installed(
    python_exec: str, dry_run: bool, logger
) -> dict[str, Any]:
    probe = run_cmd([python_exec, "-m", "pre_commit", "--version"], cwd=repo_root())
    if probe.returncode == 0:
        return {"installed": True, "version": probe.stdout}

    if dry_run:
        return {"installed": False, "action": "would-install"}

    install = run_cmd(
        [python_exec, "-m", "pip", "install", "pre-commit"], cwd=repo_root()
    )
    return {
        "installed": install.returncode == 0,
        "stdout": install.stdout,
        "stderr": install.stderr,
    }


def install_hooks(dry_run: bool) -> list[dict[str, Any]]:
    commands = [
        ["pre-commit", "install"],
        ["pre-commit", "install", "--hook-type", "commit-msg"],
        ["pre-commit", "install", "--hook-type", "pre-push"],
    ]
    results: list[dict[str, Any]] = []
    for cmd in commands:
        if dry_run:
            results.append({"command": cmd, "status": "dry-run"})
            continue
        result = run_cmd(cmd, cwd=repo_root())
        results.append(
            {
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return results


def parse_hook_results(output: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        # Typical format: "black.................................................................Passed"
        for hook in HOOK_NAMES:
            if line.lower().startswith(hook.lower()):
                status = "unknown"
                if "Passed" in line:
                    status = "passed"
                elif "Failed" in line:
                    status = "failed"
                elif "Skipped" in line:
                    status = "skipped"
                parsed[hook] = {"line": line, "status": status}

    for hook in HOOK_NAMES:
        parsed.setdefault(hook, {"line": "not-reported", "status": "not-reported"})
    return parsed


def run_all_files(
    auto_stage: bool, interactive: bool, dry_run: bool, logger
) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry-run", "hooks": {}}

    first = run_cmd(["pre-commit", "run", "--all-files"], cwd=repo_root())
    combined = (first.stdout or "") + "\n" + (first.stderr or "")
    hook_map = parse_hook_results(combined)

    modified = run_cmd(["git", "status", "--porcelain"], cwd=repo_root())
    changed_files = [
        line[3:]
        for line in modified.stdout.splitlines()
        if line and not line.startswith("??")
    ]

    failed_hooks = [
        name for name, item in hook_map.items() if item["status"] == "failed"
    ]
    auto_fix_only = failed_hooks and all(
        hook in AUTO_FIX_HOOKS for hook in failed_hooks
    )

    rerun = None
    if auto_stage and changed_files:
        run_cmd(["git", "add", "."], cwd=repo_root())

    if auto_fix_only and changed_files:
        rerun = run_cmd(["pre-commit", "run", "--all-files"], cwd=repo_root())
        hook_map = parse_hook_results(
            (rerun.stdout or "") + "\n" + (rerun.stderr or "")
        )

    if interactive and failed_hooks and not auto_fix_only:
        print("Manual intervention required for hooks:", ", ".join(failed_hooks))
        _ = input("Fix issues and press Enter to continue (or Ctrl+C to stop)...")
        rerun = run_cmd(["pre-commit", "run", "--all-files"], cwd=repo_root())
        hook_map = parse_hook_results(
            (rerun.stdout or "") + "\n" + (rerun.stderr or "")
        )

    return {
        "first_run": {
            "returncode": first.returncode,
            "stdout": first.stdout,
            "stderr": first.stderr,
        },
        "rerun": (
            None
            if rerun is None
            else {
                "returncode": rerun.returncode,
                "stdout": rerun.stdout,
                "stderr": rerun.stderr,
            }
        ),
        "hooks": hook_map,
        "changed_files": changed_files,
    }


def write_guide(root: Path) -> Path:
    guide = root / "docs" / "reports" / "precommit_usage.md"
    lines = [
        "# Pre-commit Usage Guide",
        "",
        "## Run Manually",
        "```bash\npre-commit run --all-files\n```",
        "",
        "## Emergency Skip (avoid unless absolutely necessary)",
        "```bash\ngit commit --no-verify\n```",
        "",
        "## Update Hooks",
        "```bash\npre-commit autoupdate\npre-commit run --all-files\n```",
        "",
        "## Add New Hook",
        "Edit `.pre-commit-config.yaml`, then run pre-commit install and pre-commit run --all-files.",
    ]
    write_text(guide, "\n".join(lines) + "\n")
    return guide


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install and validate pre-commit hooks"
    )
    parser.add_argument("--python", default="python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--auto-stage", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("install_precommit_hooks")
    root = repo_root()

    try:
        install_info = ensure_precommit_installed(args.python, args.dry_run, logger)
        hook_install = install_hooks(args.dry_run)
        run_report = (
            {"status": "skipped"}
            if args.skip_run
            else run_all_files(args.auto_stage, args.interactive, args.dry_run, logger)
        )
        usage = write_guide(root)

        report = {
            "timestamp": utc_now(),
            "install": install_info,
            "hook_install": hook_install,
            "run_report": run_report,
            "usage_guide": str(usage),
        }
        out_json = root / "logs" / "bootstrap" / "precommit_report.json"
        out_md = root / "logs" / "bootstrap" / "precommit_report.md"
        write_json(out_json, report)

        lines = ["# Pre-commit Report", "", f"- Generated: {utc_now()}"]
        if isinstance(run_report, dict) and run_report.get("hooks"):
            lines.append("\n## Hook Results")
            for hook, item in run_report["hooks"].items():
                lines.append(f"- {hook}: {item.get('status')}")
        write_text(out_md, "\n".join(lines) + "\n")

        print(json.dumps(report, indent=2))
        if args.dry_run:
            return 0
        failed = []
        hooks = run_report.get("hooks", {}) if isinstance(run_report, dict) else {}
        for name, item in hooks.items():
            if item.get("status") == "failed":
                failed.append(name)
        return 1 if failed else 0
    except Exception as exc:
        logger.error("Pre-commit setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
