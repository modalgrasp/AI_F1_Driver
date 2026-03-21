#!/usr/bin/env python3
"""Master bootstrap entrypoint for Step 1.5 Phase 1 finalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import repo_root, run_cmd, setup_logger, utc_now, write_json


def step(label: str, command: list[str], allow_fail: bool, dry_run: bool, logger) -> dict:
    logger.info("[%s] %s", label, " ".join(command))
    if dry_run:
        return {"label": label, "command": command, "status": "dry-run", "returncode": 0}
    result = run_cmd(command, cwd=repo_root())
    ok = result.returncode == 0 or allow_fail
    return {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "ok": ok,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end repository bootstrap runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--user-name", default="USER_NAME_PLACEHOLDER")
    parser.add_argument("--user-email", default="USER_EMAIL_PLACEHOLDER")
    parser.add_argument("--remote-url", default=None)
    parser.add_argument("--remote-platform", default=None)
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--skip-commit", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("bootstrap_repository")
    root = repo_root()

    commands = [
        (
            "1/10 prepare initial commit",
            [sys.executable, "scripts/prepare_initial_commit.py", "--output-dir", "logs/bootstrap"]
            + (["--dry-run"] if args.dry_run else [])
            + (["--interactive"] if args.interactive else []),
            False,
        ),
    ]

    if not args.skip_commit:
        commands.append(
            (
                "2/10 create initial commit",
                [
                    sys.executable,
                    "scripts/create_initial_commit.py",
                    "--user-name",
                    args.user_name,
                    "--user-email",
                    args.user_email,
                ]
                + (["--dry-run"] if args.dry_run else [])
                + (["--interactive"] if args.interactive else []),
                False,
            )
        )

    if not args.skip_remote:
        remote_cmd = [
            sys.executable,
            "scripts/setup_remote_repository.py",
            "--remote",
            "origin",
        ]
        if args.remote_url:
            remote_cmd.extend(["--remote-url", args.remote_url])
        if args.remote_platform:
            remote_cmd.extend(["--platform", args.remote_platform])
        if args.github_token:
            remote_cmd.extend(["--github-token", args.github_token])
        if args.interactive and not args.remote_url:
            remote_cmd.append("--interactive")
        if args.dry_run:
            remote_cmd.append("--dry-run")
        commands.append(("3/10 setup remote", remote_cmd, False))

    commands.extend(
        [
            (
                "4/10 install pre-commit hooks",
                [sys.executable, "scripts/install_precommit_hooks.py", "--auto-stage"]
                + (["--dry-run"] if args.dry_run else [])
                + (["--interactive"] if args.interactive else []),
                False,
            ),
            (
                "5/10 run phase1 tests",
                [sys.executable, "scripts/run_phase1_tests.py", "--output-dir", "logs/bootstrap"],
                True,
            ),
            (
                "6/10 generate phase1 report",
                [sys.executable, "scripts/generate_phase1_report.py", "--output-dir", "docs/reports"],
                False,
            ),
            (
                "7/10 repository health check",
                [sys.executable, "scripts/check_repository_health.py", "--output-dir", "logs/bootstrap", "--quick"],
                True,
            ),
            (
                "8/10 validate dev environment",
                [sys.executable, "scripts/validate_dev_environment.py", "--output-dir", "logs/bootstrap"],
                True,
            ),
            (
                "9/10 update bootstrap docs",
                [sys.executable, "scripts/update_bootstrap_docs.py"]
                + (["--repo-url", args.remote_url] if args.remote_url else []),
                False,
            ),
            (
                "10/10 post-bootstrap validation",
                [sys.executable, "scripts/post_bootstrap_validation.py", "--create-marker"],
                True,
            ),
        ]
    )

    results = []
    failed_hard = False
    for label, command, allow_fail in commands:
        result = step(label, command, allow_fail, args.dry_run, logger)
        results.append(result)
        if not args.dry_run and result.get("returncode", 0) != 0 and not allow_fail:
            failed_hard = True
            logger.error("Stopping bootstrap at %s", label)
            break

    summary = {
        "timestamp": utc_now(),
        "dry_run": args.dry_run,
        "failed_hard": failed_hard,
        "results": results,
    }
    out = root / "logs" / "bootstrap" / "bootstrap_summary.json"
    write_json(out, summary)

    print(json.dumps(summary, indent=2))
    if failed_hard:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
