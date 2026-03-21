#!/usr/bin/env python3
"""Training entrypoint."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.json")
    args = parser.parse_args()
    print(f"TODO: training with {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
