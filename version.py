#!/usr/bin/env python3
"""Semantic version management utility."""

from __future__ import annotations

import argparse
from pathlib import Path

__version__ = "0.1.0"


def bump(version: str, part: str) -> str:
    major, minor, patch = [int(x) for x in version.split(".")]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def update_version_file(new_version: str) -> None:
    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    updated = text.replace(f'__version__ = "{__version__}"', f'__version__ = "{new_version}"')
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Version manager")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default=None)
    args = parser.parse_args()

    if args.bump:
        new_version = bump(__version__, args.bump)
        update_version_file(new_version)
        print(new_version)
    else:
        print(__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
