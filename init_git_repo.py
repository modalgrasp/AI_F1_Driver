#!/usr/bin/env python3
"""Initialize and configure Git repository for F1 Racing AI."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


def run(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def ensure_git_installed() -> None:
    if shutil.which("git") is None:
        raise RuntimeError(
            "Git not found. Install Git from https://git-scm.com/downloads"
        )


def configure_git(root: Path, user_name: str, user_email: str) -> None:
    commands = [
        ["git", "config", "user.name", user_name],
        ["git", "config", "user.email", user_email],
        ["git", "config", "init.defaultBranch", "main"],
        ["git", "config", "core.autocrlf", "input" if os.name != "nt" else "true"],
        ["git", "config", "pull.rebase", "false"],
    ]
    for cmd in commands:
        code, out = run(cmd, root)
        if code != 0:
            raise RuntimeError(f"Failed command {' '.join(cmd)}: {out}")


def init_repo(root: Path) -> None:
    if not (root / ".git").exists():
        code, out = run(["git", "init", "-b", "main"], root)
        if code != 0:
            raise RuntimeError(f"git init failed: {out}")
        LOGGER.info("Repository initialized")
    else:
        LOGGER.info("Repository already initialized")


def create_hooks(root: Path) -> None:
    hooks = root / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)

    pre_commit = hooks / "pre-commit"
    pre_push = hooks / "pre-push"
    commit_msg = hooks / "commit-msg"

    pre_commit.write_text(
        "#!/usr/bin/env bash\n"
        "echo '[hook] pre-commit: black/flake8/quick tests'\n"
        "python -m black --check . || exit 1\n"
        "python -m flake8 . || exit 1\n"
        "python -m pytest -m 'not slow' -q || exit 1\n",
        encoding="utf-8",
    )

    pre_push.write_text(
        "#!/usr/bin/env bash\n"
        "echo '[hook] pre-push: full tests'\n"
        "python -m pytest -q || exit 1\n",
        encoding="utf-8",
    )

    commit_msg.write_text(
        "#!/usr/bin/env bash\n"
        "msg_file=$1\n"
        "pattern='^(feat|fix|docs|style|refactor|test|chore)\\([a-zA-Z0-9_-]+\\): .+'\n"
        'if ! grep -Eq "$pattern" "$msg_file"; then\n'
        "  echo 'Commit message must match <type>(<scope>): <subject>'\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )

    if os.name != "nt":
        for hook in [pre_commit, pre_push, commit_msg]:
            hook.chmod(0o755)


def initial_commit(root: Path) -> None:
    run(["git", "add", "."], root)
    code, out = run(
        ["git", "commit", "-m", "chore(repo): initialize project structure"], root
    )
    if code != 0 and "nothing to commit" not in out.lower():
        raise RuntimeError(f"Initial commit failed: {out}")


def configure_remote(root: Path, remote_url: str | None) -> None:
    if not remote_url:
        return
    run(["git", "remote", "remove", "origin"], root)
    code, out = run(["git", "remote", "add", "origin", remote_url], root)
    if code != 0:
        raise RuntimeError(f"Failed to add remote: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Git repository and hooks")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--user-name", default=None)
    parser.add_argument("--user-email", default=None)
    parser.add_argument("--remote-url", default=None)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    setup_logging(Path("logs"), level="INFO", console=True)
    root = args.root.resolve()

    ensure_git_installed()
    init_repo(root)

    user_name = args.user_name or input("Git user.name: ").strip()
    user_email = args.user_email or input("Git user.email: ").strip()
    configure_git(root, user_name, user_email)

    create_hooks(root)
    configure_remote(root, args.remote_url)
    initial_commit(root)

    if args.push and args.remote_url:
        code, out = run(["git", "push", "-u", "origin", "main"], root)
        if code != 0:
            raise RuntimeError(f"git push failed: {out}")

    LOGGER.info("Git repository setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
