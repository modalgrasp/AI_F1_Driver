#!/usr/bin/env python3
"""Package setup for f1-racing-ai."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def load_requirements(path: str) -> list[str]:
    req_path = ROOT / path
    lines = req_path.read_text(encoding="utf-8").splitlines()
    deps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        deps.append(line)
    return deps


setup(
    name="f1-racing-ai",
    version="0.1.0",
    description="Autonomous F1 racing AI with Assetto Corsa and Gymnasium",
    author="F1 Autonomous Racing AI Team",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "experiments"]),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "f1-dev-tools=dev_tools:main",
            "f1-validate-structure=validate_project_structure:main",
            "f1-install-torch=install_pytorch:main",
            "f1-track-install=track_installer:main",
            "f1-track-validate=track_validator:main",
            "f1-gpu-test=test_gpu_setup:main",
        ]
    },
)
