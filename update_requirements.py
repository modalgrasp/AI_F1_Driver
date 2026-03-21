#!/usr/bin/env python3
"""Check and update dependency versions safely."""

from __future__ import annotations

import subprocess


def main() -> int:
    print("Checking outdated packages...")
    subprocess.run(["python", "-m", "pip", "list", "--outdated"], check=False)
    print("Tip: update one package at a time, then run tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
