#!/usr/bin/env python3
"""Developer tooling CLI for formatting, linting, testing, docs, clean, backup."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def format_code() -> int:
    return max(run(["python", "-m", "black", "."]), run(["python", "-m", "isort", "."]))


def lint_code() -> int:
    a = run(["python", "-m", "flake8", "."])
    b = run(["python", "-m", "mypy", "."])
    c = run(["python", "-m", "pylint", "utils", "environments", "training"])
    return max(a, b, c)


def run_tests(marker: str | None = None) -> int:
    cmd = ["python", "-m", "pytest", "-q"]
    if marker:
        cmd.extend(["-m", marker])
    return run(cmd)


def build_docs() -> int:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    return run(["python", "-m", "mkdocs", "build"])


def clean() -> int:
    for pattern in ["__pycache__", ".pytest_cache", ".mypy_cache", "htmlcov"]:
        for path in Path(".").rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    return 0


def backup(target: Path = Path("backups")) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for src in [Path("configs"), Path("models"), Path("experiments")]:
        if src.exists():
            shutil.make_archive(str(target / src.name), "zip", root_dir=str(src))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Developer utilities")
    parser.add_argument("command", choices=["format", "lint", "test", "docs", "clean", "backup"])
    parser.add_argument("--marker", default=None)
    args = parser.parse_args()

    if args.command == "format":
        return format_code()
    if args.command == "lint":
        return lint_code()
    if args.command == "test":
        return run_tests(args.marker)
    if args.command == "docs":
        return build_docs()
    if args.command == "clean":
        return clean()
    if args.command == "backup":
        return backup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
