#!/usr/bin/env python3
"""Configure and validate remote repository settings.

Supports GitHub/GitLab/Bitbucket/custom remotes, origin updates, push setup,
and optional GitHub branch protection when token is available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
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

URL_RE = re.compile(
    r"^(?:git@[^:]+:[^\s]+\.git|https://[^\s]+(?:\.git)?)$",
    re.IGNORECASE,
)


def validate_url(url: str) -> bool:
    return bool(URL_RE.match(url.strip()))


def detect_platform(url: str) -> str:
    lower = url.lower()
    if "github" in lower:
        return "github"
    if "gitlab" in lower:
        return "gitlab"
    if "bitbucket" in lower:
        return "bitbucket"
    return "custom"


def parse_github_repo(url: str) -> tuple[str, str] | None:
    # Supports git@github.com:owner/repo.git and https://github.com/owner/repo(.git)
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)
    https_match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if https_match:
        return https_match.group(1), https_match.group(2)
    return None


def set_remote(
    root: Path, remote: str, url: str, dry_run: bool, logger
) -> dict[str, Any]:
    existing = run_cmd(["git", "remote"], cwd=root)
    remotes = existing.stdout.splitlines() if existing.returncode == 0 else []

    actions: list[str] = []
    if remote in remotes:
        actions.append(f"update {remote}")
        if not dry_run:
            run_cmd(["git", "remote", "set-url", remote, url], cwd=root, check=True)
    else:
        actions.append(f"add {remote}")
        if not dry_run:
            run_cmd(["git", "remote", "add", remote, url], cwd=root, check=True)

    connectivity_ok = False
    if not dry_run:
        probe = run_cmd(["git", "ls-remote", remote], cwd=root)
        connectivity_ok = probe.returncode == 0

    logger.info("Remote %s configured (%s)", remote, ", ".join(actions))
    return {"remote": remote, "actions": actions, "connectivity_ok": connectivity_ok}


def configure_git_defaults(root: Path, dry_run: bool) -> None:
    commands = [
        ["git", "config", "push.default", "simple"],
        ["git", "config", "pull.rebase", "true"],
        ["git", "config", "fetch.prune", "true"],
    ]
    if not dry_run:
        for cmd in commands:
            run_cmd(cmd, cwd=root, check=True)


def maybe_configure_credentials(root: Path, url: str, dry_run: bool, logger) -> str:
    if url.startswith("https://"):
        if dry_run:
            return "dry-run"
        # Windows Git Credential Manager is standard; use cache helper as fallback on other OS.
        if Path(
            "C:/Program Files/Git/mingw64/libexec/git-core/git-credential-manager-core.exe"
        ).exists():
            run_cmd(["git", "config", "credential.helper", "manager-core"], cwd=root)
            return "manager-core"
        run_cmd(
            ["git", "config", "credential.helper", "cache --timeout=3600"], cwd=root
        )
        return "cache"
    logger.info("SSH remote detected; credential helper unchanged.")
    return "ssh"


def push_initial(
    root: Path, remote: str, branch: str, dry_run: bool, logger
) -> dict[str, Any]:
    if dry_run:
        return {"branch_push": "dry-run", "tags_push": "dry-run"}

    branch_push = run_cmd(["git", "push", "-u", remote, branch], cwd=root)
    tags_push = run_cmd(["git", "push", remote, "--tags"], cwd=root)
    if branch_push.returncode != 0:
        logger.warning(
            "Branch push failed: %s", branch_push.stderr or branch_push.stdout
        )
    if tags_push.returncode != 0:
        logger.warning("Tag push failed: %s", tags_push.stderr or tags_push.stdout)

    return {
        "branch_push": branch_push.returncode,
        "branch_push_output": branch_push.stdout or branch_push.stderr,
        "tags_push": tags_push.returncode,
        "tags_push_output": tags_push.stdout or tags_push.stderr,
    }


def github_branch_protection(
    url: str, token: str | None, dry_run: bool, logger
) -> dict[str, Any]:
    info = {"applied": False, "reason": "not-applicable"}
    parsed = parse_github_repo(url)
    if not parsed:
        info["reason"] = "not-github"
        return info
    if not token:
        info["reason"] = "missing-token"
        return info

    owner, repo = parsed
    endpoint = f"https://api.github.com/repos/{owner}/{repo}/branches/main/protection"
    payload = {
        "required_status_checks": {
            "strict": True,
            "contexts": ["test", "lint"],
        },
        "enforce_admins": False,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
        },
        "restrictions": None,
        "allow_force_pushes": False,
        "allow_deletions": False,
    }

    if dry_run:
        return {"applied": False, "reason": "dry-run", "endpoint": endpoint}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="PUT")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8", errors="ignore")
            logger.info("GitHub branch protection response: %s", response.status)
            return {
                "applied": response.status in {200, 201},
                "status": response.status,
                "body": body[:400],
            }
    except urllib.error.HTTPError as exc:
        return {
            "applied": False,
            "reason": f"http-{exc.code}",
            "body": exc.read().decode("utf-8", errors="ignore")[:400],
        }
    except Exception as exc:
        return {"applied": False, "reason": str(exc)}


def write_access_guide(root: Path, remote_url: str, branch: str) -> Path:
    path = root / "docs" / "reports" / "repository_access.md"
    lines = [
        "# Repository Access Instructions",
        "",
        f"- Remote URL: {remote_url}",
        f"- Default branch: {branch}",
        "",
        "## Clone",
        f"```bash\ngit clone {remote_url}\ncd F1\ngit checkout {branch}\n```",
        "",
        "## SSH Setup",
        '```bash\nssh-keygen -t ed25519 -C "your_email@example.com"\n# add public key to your git provider\n```',
        "",
        "## HTTPS Token Setup",
        "Create a personal access token with repository scope and use it as password when prompted.",
    ]
    write_text(path, "\n".join(lines) + "\n")
    return path


def write_badges(root: Path, platform: str, remote_url: str) -> Path:
    path = root / "docs" / "reports" / "repository_badges.md"
    if platform == "github":
        repo = parse_github_repo(remote_url)
        slug = f"{repo[0]}/{repo[1]}" if repo else "owner/repo"
        badges = [
            f"![Build](https://github.com/{slug}/actions/workflows/tests.yml/badge.svg)",
            "![License](https://img.shields.io/badge/license-MIT-green)",
            "![Python](https://img.shields.io/badge/python-3.10%2B-yellow)",
            "![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)",
            "![Docs](https://img.shields.io/badge/docs-mkdocs-blue)",
        ]
    else:
        badges = [
            "![Build](https://img.shields.io/badge/build-pending-lightgrey)",
            "![License](https://img.shields.io/badge/license-MIT-green)",
            "![Python](https://img.shields.io/badge/python-3.10%2B-yellow)",
        ]
    write_text(path, "\n".join(badges) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up and validate repository remote configuration"
    )
    parser.add_argument(
        "--platform", choices=["github", "gitlab", "bitbucket", "custom"], default=None
    )
    parser.add_argument("--remote-url", default=None)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--visibility", choices=["public", "private"], default="private"
    )
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("setup_remote_repository")
    root = repo_root()

    remote_url = args.remote_url
    if args.interactive and not remote_url:
        remote_url = input("Remote repository URL (SSH/HTTPS): ").strip()

    if not remote_url:
        logger.error("Remote URL is required.")
        return 1
    if not validate_url(remote_url):
        logger.error("Invalid remote URL format: %s", remote_url)
        return 1

    platform_name = args.platform or detect_platform(remote_url)
    logger.info("Configuring remote '%s' on platform %s", args.remote, platform_name)

    try:
        remote_info = set_remote(root, args.remote, remote_url, args.dry_run, logger)
        configure_git_defaults(root, args.dry_run)
        credential_helper = maybe_configure_credentials(
            root, remote_url, args.dry_run, logger
        )
        push_info = push_initial(root, args.remote, args.branch, args.dry_run, logger)
        protection = github_branch_protection(
            remote_url,
            token=args.github_token,
            dry_run=args.dry_run,
            logger=logger,
        )

        access_guide = write_access_guide(root, remote_url, args.branch)
        badges = write_badges(root, platform_name, remote_url)

        report = {
            "timestamp": utc_now(),
            "platform": platform_name,
            "visibility": args.visibility,
            "remote": args.remote,
            "remote_url": remote_url,
            "remote_info": remote_info,
            "credential_helper": credential_helper,
            "push": push_info,
            "branch_protection": protection,
            "artifacts": {
                "access_guide": str(access_guide),
                "badges": str(badges),
            },
        }
        out = root / "logs" / "bootstrap" / "remote_setup_report.json"
        write_json(out, report)
        logger.info("Wrote report: %s", out)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        logger.error("Remote setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
