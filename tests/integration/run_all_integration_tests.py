#!/usr/bin/env python3
"""Run all Phase 2 integration and performance tests."""

from __future__ import annotations

import sys

import pytest


def main() -> int:
    print("=" * 70)
    print("PHASE 2 INTEGRATION TESTS")
    print("=" * 70)
    print()

    test_files = [
        "tests/integration/test_full_system.py",
        "tests/performance/test_performance_regression.py",
    ]

    exit_code = pytest.main(
        [
            "-v",
            "--tb=short",
            "--maxfail=3",
            *test_files,
        ]
    )

    print()
    print("=" * 70)
    if exit_code == 0:
        print("ALL INTEGRATION TESTS PASSED")
        print("Phase 2 Complete. Ready for Phase 3.")
    else:
        print("SOME INTEGRATION TESTS FAILED")
        print("Fix failures before proceeding to Phase 3.")
    print("=" * 70)
    print()

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
