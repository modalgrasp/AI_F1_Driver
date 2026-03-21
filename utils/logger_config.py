#!/usr/bin/env python3
"""Centralized logging and diagnostics configuration."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import importlib
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import psutil

torch = None
GPUtil = None

try:
    torch = importlib.import_module("torch")
except Exception:  # pragma: no cover - optional during bootstrap
    torch = None

try:
    GPUtil = importlib.import_module("GPUtil")
except Exception:  # pragma: no cover - optional diagnostics
    GPUtil = None


def setup_logging(log_dir: str | Path, level: str = "INFO", console: bool = True) -> logging.Logger:
    """Configure application-wide logging handlers.

    Args:
        log_dir: Directory where log files are stored.
        level: Root log level string.
        console: Whether to enable console logging.

    Returns:
        The configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicate logs on repeated setup calls.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_specs = {
        "application.log": logging.INFO,
        "environment.log": logging.DEBUG,
        "errors.log": logging.ERROR,
        "performance.log": logging.INFO,
    }

    for filename, min_level in file_specs.items():
        file_handler = RotatingFileHandler(
            log_path / filename,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(min_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(root_logger.level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    root_logger.info("Logging configured. Directory: %s", log_path)
    return root_logger


def get_system_info() -> dict[str, Any]:
    """Collect CPU/GPU/RAM/platform information for diagnostics."""
    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
    }

    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            info["gpus"] = [
                {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_free_mb": gpu.memoryFree,
                    "driver": gpu.driver,
                }
                for gpu in gpus
            ]
        except Exception as exc:  # pragma: no cover
            info["gpus"] = [{"error": str(exc)}]
    else:
        info["gpus"] = []

    return info


def check_cuda_availability() -> dict[str, Any]:
    """Run CUDA/cudNN diagnostics via PyTorch."""
    result: dict[str, Any] = {
        "torch_installed": torch is not None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_available": False,
        "device_name": None,
        "vram_gb": None,
    }

    if torch is None:
        return result

    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_version"] = torch.version.cuda
    result["cudnn_available"] = bool(torch.backends.cudnn.is_available())

    if result["cuda_available"]:
        device_index = torch.cuda.current_device()
        result["device_name"] = torch.cuda.get_device_name(device_index)
        vram_bytes = torch.cuda.get_device_properties(device_index).total_memory
        result["vram_gb"] = round(vram_bytes / (1024**3), 2)

    return result


def check_disk_space(path: str | Path) -> dict[str, float]:
    """Return disk space information for a target directory path."""
    usage = shutil.disk_usage(Path(path))
    return {
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def verify_directory_structure(root_dir: str | Path) -> dict[str, bool]:
    """Verify expected project directories exist."""
    root = Path(root_dir)
    expected = [
        "environments",
        "utils",
        "tests",
        "logs",
        "data",
        "models",
        "configs",
        "docs",
    ]
    return {name: (root / name).exists() for name in expected}


def detect_compute_capability() -> str | None:
    """Detect CUDA compute capability from the active GPU if available."""
    if torch is None or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def environment_path_hint() -> str:
    """Return a short text with likely virtual environment path for current OS."""
    if os.name == "nt":
        return "f1_racing_env\\Scripts\\activate"
    return "source f1_racing_env/bin/activate"
