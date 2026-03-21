#!/usr/bin/env python3
"""Comprehensive environment validation script for F1 racing project."""

from __future__ import annotations

import importlib
import platform
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    _colorama = importlib.import_module("colorama")
    Fore = _colorama.Fore
    Style = _colorama.Style
    init = _colorama.init
except Exception:  # pragma: no cover - fallback when colorama is missing

    class _DummyColor:
        RED = ""
        GREEN = ""
        YELLOW = ""

    class _DummyStyle:
        RESET_ALL = ""

    Fore = _DummyColor()  # type: ignore[assignment]
    Style = _DummyStyle()  # type: ignore[assignment]

    def init(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


from environments.assetto_corsa_connector import AssettoCorsa_Connector
from environments.f1_racing_env import F1RacingEnv
from utils.config_manager import ConfigError, ConfigManager
from utils.logger_config import (
    check_cuda_availability,
    detect_compute_capability,
    get_system_info,
)

init(autoreset=True)

REQUIRED_PACKAGES = [
    "gymnasium",
    "numpy",
    "pandas",
    "torch",
    "matplotlib",
    "plotly",
    "psutil",
]


class ValidationRunner:
    """Runs all validation checks and writes a report."""

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent
        self.results: list[tuple[str, bool, str]] = []

    def add_result(self, name: str, success: bool, message: str) -> None:
        self.results.append((name, success, message))
        color = Fore.GREEN if success else Fore.RED
        prefix = "PASS" if success else "FAIL"
        if "WARN" in message.upper() and success:
            color = Fore.YELLOW
            prefix = "WARN"
        print(color + f"[{prefix}] {name}: {message}" + Style.RESET_ALL)

    def run(self) -> int:
        self.check_python_version()
        self.check_required_packages()
        self.check_custom_imports()
        self.check_cuda()
        self.check_gpu_memory()
        self.check_assetto_installation()
        self.check_shared_memory()
        self.check_config()
        self.check_environment_creation()
        self.run_quick_sanity_episode()

        success = all(item[1] for item in self.results)
        self.write_report(success)
        return 0 if success else 1

    def check_python_version(self) -> None:
        ok = sys.version_info >= (3, 10)
        self.add_result(
            "Python Version",
            ok,
            f"Detected {platform.python_version()} (required >= 3.10)",
        )

    def check_required_packages(self) -> None:
        missing: list[str] = []
        for package in REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
            except Exception:
                missing.append(package)

        sb3_needed = sys.version_info < (3, 14)
        if sb3_needed:
            try:
                importlib.import_module("stable_baselines3")
            except Exception:
                missing.append("stable_baselines3")

        self.add_result(
            "Required Packages",
            not missing,
            (
                "All packages available"
                if not missing
                else f"Missing packages: {', '.join(missing)}"
            ),
        )

        if not sb3_needed:
            self.add_result(
                "Stable-Baselines3 Compatibility",
                True,
                "WARN: Python 3.14 detected; SB3 may be unavailable. Use Python 3.11/3.12 for training.",
            )

    def check_custom_imports(self) -> None:
        try:
            importlib.import_module("utils.shared_memory_reader")
            importlib.import_module("utils.config_manager")
            importlib.import_module("utils.logger_config")
            importlib.import_module("environments.assetto_corsa_connector")
            importlib.import_module("environments.f1_racing_env")
            self.add_result(
                "Custom Module Imports", True, "Custom modules import successfully"
            )
        except Exception as exc:
            self.add_result("Custom Module Imports", False, f"Import failed: {exc}")

    def check_cuda(self) -> None:
        data = check_cuda_availability()
        ok = bool(data.get("torch_installed")) and bool(data.get("cuda_available"))
        msg = (
            f"cuda_available={data.get('cuda_available')}, cuda_version={data.get('cuda_version')}, "
            f"cudnn_available={data.get('cudnn_available')}"
        )
        self.add_result("CUDA Availability", ok, msg)

    def check_gpu_memory(self) -> None:
        data = check_cuda_availability()
        vram = data.get("vram_gb")
        capability = detect_compute_capability()
        ok = vram is not None
        self.add_result(
            "GPU Memory", ok, f"vram_gb={vram}, compute_capability={capability}"
        )

    def check_assetto_installation(self) -> None:
        try:
            connector = AssettoCorsa_Connector(
                config_path=self.root / "configs/config.json"
            )
            connector.validate_installation()
            self.add_result(
                "Assetto Corsa Install", True, f"Found at {connector.install_path}"
            )
        except Exception as exc:
            self.add_result("Assetto Corsa Install", False, str(exc))

    def check_shared_memory(self) -> None:
        try:
            connector = AssettoCorsa_Connector(
                config_path=self.root / "configs/config.json"
            )
            connector.connect_shared_memory()
            state = connector.read_state()
            connector.disconnect_shared_memory()
            self.add_result(
                "Shared Memory Access",
                True,
                f"Read state keys: {sorted(state.keys())[:4]}...",
            )
        except Exception as exc:
            self.add_result(
                "Shared Memory Access",
                False,
                f"{exc}. Launch Assetto Corsa and start a session before retrying.",
            )

    def check_config(self) -> None:
        try:
            cm = ConfigManager(self.root / "configs/config.json")
            cm.load()
            self.add_result("Configuration", True, "Config file loaded and validated")
        except ConfigError as exc:
            self.add_result("Configuration", False, str(exc))

    def check_environment_creation(self) -> None:
        try:
            env = F1RacingEnv(config_path=str(self.root / "configs/config.json"))
            obs, _ = env.reset(seed=123)
            env.close()
            self.add_result(
                "Environment Creation", True, f"Observation shape: {obs.shape}"
            )
        except Exception as exc:
            self.add_result("Environment Creation", False, str(exc))

    def run_quick_sanity_episode(self) -> None:
        try:
            env = F1RacingEnv(config_path=str(self.root / "configs/config.json"))
            obs, _ = env.reset(seed=7)
            total_reward = 0.0
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            env.close()
            self.add_result(
                "Sanity Episode",
                True,
                f"Ran random policy, total_reward={total_reward:.3f}",
            )
        except Exception as exc:
            self.add_result("Sanity Episode", False, str(exc))

    def write_report(self, success: bool) -> None:
        report_path = self.root / "logs" / "validation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "F1 Autonomous Racing AI - Validation Report",
            f"Timestamp: {datetime.now(UTC).isoformat()}",
            f"Overall Success: {success}",
            "",
            "System Info:",
            str(get_system_info()),
            "",
            "Check Results:",
        ]
        for name, ok, message in self.results:
            lines.append(f"- {name}: {'PASS' if ok else 'FAIL'} | {message}")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(
            Fore.YELLOW
            + f"Validation report written to: {report_path}"
            + Style.RESET_ALL
        )


def main() -> int:
    """Entrypoint for setup validation."""
    runner = ValidationRunner()
    try:
        return runner.run()
    except Exception:  # pragma: no cover - top-level safety net
        print(Fore.RED + "[FAIL] Unexpected validation crash" + Style.RESET_ALL)
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
