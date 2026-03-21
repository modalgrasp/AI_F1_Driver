#!/usr/bin/env python3
"""Comprehensive Phase 1 validation suite (25 checks)."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import (
    human_size,
    repo_root,
    run_cmd,
    setup_logger,
    utc_now,
    write_json,
    write_text,
)


@dataclass
class TestCaseResult:
    id: int
    name: str
    passed: bool
    critical: bool
    duration_ms: float
    message: str


class Phase1TestRunner:
    """Runs 25 test checks grouped by environment, track, GPU, repo, and integration."""

    def __init__(self, subset: str = "all") -> None:
        self.root = repo_root()
        self.subset = subset
        self.logger = setup_logger("run_phase1_tests")
        self.results: list[TestCaseResult] = []

    def run_case(
        self, idx: int, name: str, critical: bool, fn: Callable[[], tuple[bool, str]]
    ) -> None:
        t0 = time.perf_counter()
        try:
            passed, message = fn()
        except Exception as exc:
            passed, message = False, f"exception: {exc}"
        dt = (time.perf_counter() - t0) * 1000.0
        self.results.append(TestCaseResult(idx, name, passed, critical, dt, message))
        status = "PASS" if passed else "FAIL"
        self.logger.info("[%02d] %s %s - %s", idx, status, name, message)

    def _python_version(self) -> tuple[bool, str]:
        ok = sys.version_info >= (3, 10)
        return ok, f"python={platform.python_version()}"

    def _venv_active(self) -> tuple[bool, str]:
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        return in_venv, f"sys.prefix={sys.prefix}"

    def _requirements_installed(self) -> tuple[bool, str]:
        result = run_cmd([sys.executable, "-m", "pip", "check"], cwd=self.root)
        return result.returncode == 0, result.stdout or result.stderr or "pip check"

    def _assetto_detect(self) -> tuple[bool, str]:
        cfg = self.root / "configs" / "config.json"
        if not cfg.exists():
            return False, "config missing"
        data = json.loads(cfg.read_text(encoding="utf-8"))
        path = Path(data.get("assetto_corsa", {}).get("install_path", ""))
        ok = path.exists() and (path / "content" / "tracks").exists()
        return ok, str(path)

    def _gym_env_creation(self) -> tuple[bool, str]:
        mod = importlib.import_module("environments.f1_racing_env")
        env = getattr(mod, "F1RacingEnv")(config_path="configs/config.json")
        obs, _ = env.reset(seed=123)
        shape_ok = tuple(obs.shape) == (20,)
        env.close()
        return shape_ok, f"obs_shape={getattr(obs, 'shape', None)}"

    def _random_episode(self) -> tuple[bool, str]:
        result = run_cmd([sys.executable, "tests/test_environment.py"], cwd=self.root)
        return result.returncode == 0, (result.stdout or result.stderr)[-300:]

    def _yas_install(self) -> tuple[bool, str]:
        track_root = Path(
            "D:/SteamLibrary/steamapps/common/assettocorsa/content/tracks/acu_yasmarina"
        )
        if not track_root.exists():
            # fallback to config-based path
            cfg = json.loads(
                (self.root / "configs" / "config.json").read_text(encoding="utf-8")
            )
            ac = Path(cfg["assetto_corsa"]["install_path"])
            candidates = list((ac / "content" / "tracks").glob("*yas*"))
            ok = bool(candidates)
            return ok, f"candidates={len(candidates)}"
        return True, str(track_root)

    def _track_data(self) -> tuple[bool, str]:
        path = (
            self.root
            / "data"
            / "tracks"
            / "yas_marina"
            / "extracted"
            / "yas_marina_track_data.json"
        )
        if not path.exists():
            return False, "track data json missing"
        data = json.loads(path.read_text(encoding="utf-8"))
        wp = len(data.get("waypoints", []))
        boundaries = data.get("boundaries", {})
        ok = (
            wp >= 1000
            and bool(boundaries.get("left"))
            and bool(boundaries.get("right"))
        )
        return ok, f"waypoints={wp}"

    def _track_visualization(self) -> tuple[bool, str]:
        out = (
            self.root
            / "data"
            / "tracks"
            / "yas_marina"
            / "visualizations"
            / "yas_marina_layout_2d.png"
        )
        if out.exists():
            return True, str(out)
        result = run_cmd(
            [sys.executable, "track_visualizer.py", "--track-id", "yas_marina"],
            cwd=self.root,
        )
        return out.exists(), result.stdout or result.stderr

    def _track_integration(self) -> tuple[bool, str]:
        path = (
            self.root
            / "data"
            / "tracks"
            / "yas_marina"
            / "integration"
            / "integration_report.json"
        )
        if not path.exists():
            result = run_cmd(
                [
                    sys.executable,
                    "integrate_track_with_environment.py",
                    "--track-id",
                    "yas_marina",
                    "--episodes",
                    "1",
                ],
                cwd=self.root,
            )
            return result.returncode == 0, result.stdout or result.stderr
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok = bool(payload.get("episodes"))
        return ok, f"episodes={len(payload.get('episodes', []))}"

    def _cuda_available(self) -> tuple[bool, str]:
        result = run_cmd(
            [
                sys.executable,
                "-c",
                "import torch;print(torch.cuda.is_available(), torch.version.cuda, torch.backends.cudnn.is_available())",
            ],
            cwd=self.root,
        )
        ok = result.returncode == 0 and "True" in result.stdout.split()[0]
        return ok, result.stdout or result.stderr

    def _gpu_detect(self) -> tuple[bool, str]:
        result = run_cmd(
            [
                sys.executable,
                "-c",
                "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none');print(round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2) if torch.cuda.is_available() else 0)",
            ],
            cwd=self.root,
        )
        ok = result.returncode == 0 and (
            "5070" in result.stdout or "RTX" in result.stdout
        )
        return ok, result.stdout or result.stderr

    def _pytorch_gpu_ops(self) -> tuple[bool, str]:
        code = "import torch;assert torch.cuda.is_available();a=torch.randn((512,512),device='cuda');b=torch.randn((512,512),device='cuda');c=a@b;torch.cuda.synchronize();print(float(c.mean()))"
        result = run_cmd([sys.executable, "-c", code], cwd=self.root)
        return result.returncode == 0, result.stdout or result.stderr

    def _mixed_precision(self) -> tuple[bool, str]:
        result = run_cmd(
            [
                sys.executable,
                "-c",
                "import torch;assert torch.cuda.is_available();x=torch.randn((256,256),device='cuda');w=torch.randn((256,256),device='cuda');\nwith torch.autocast('cuda',dtype=torch.float16):y=x@w\nprint(y.dtype)",
            ],
            cwd=self.root,
        )
        return (
            result.returncode == 0 and "float16" in result.stdout,
            result.stdout or result.stderr,
        )

    def _memory_management(self) -> tuple[bool, str]:
        code = "import torch;assert torch.cuda.is_available();x=torch.empty((1024,1024,8),device='cuda');del x;torch.cuda.empty_cache();print('ok')"
        result = run_cmd([sys.executable, "-c", code], cwd=self.root)
        return result.returncode == 0, result.stdout or result.stderr

    def _batch_opt(self) -> tuple[bool, str]:
        result = run_cmd(
            [
                sys.executable,
                "batch_size_optimizer.py",
                "--min-batch",
                "32",
                "--max-batch",
                "1024",
            ],
            cwd=self.root,
        )
        return result.returncode == 0, (result.stdout or result.stderr)[-300:]

    def _git_repo(self) -> tuple[bool, str]:
        git_dir = self.root / ".git"
        status = run_cmd(["git", "status", "--porcelain"], cwd=self.root)
        return git_dir.exists() and status.returncode == 0, "git status ok"

    def _project_structure(self) -> tuple[bool, str]:
        result = run_cmd(
            [sys.executable, "validate_project_structure.py"], cwd=self.root
        )
        ok = result.returncode == 0 and '"ok": true' in (result.stdout or "").lower()
        return ok, (result.stdout or result.stderr)[-400:]

    def _config_files(self) -> tuple[bool, str]:
        import yaml  # type: ignore

        # JSON
        for path in (self.root / "configs").glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))
        # YAML
        for path in self.root.rglob("*.yml"):
            yaml.safe_load(path.read_text(encoding="utf-8"))
        return True, "json/yaml valid"

    def _import_tests(self) -> tuple[bool, str]:
        modules = [
            "utils.config_manager",
            "utils.shared_memory_reader",
            "environments.f1_racing_env",
            "training.algorithms.ppo",
            "control.steering_controller",
            "planning.racing_line",
            "vehicle_dynamics.tire_model",
        ]
        for name in modules:
            importlib.import_module(name)
        return True, f"imported {len(modules)} modules"

    def _documentation(self) -> tuple[bool, str]:
        required = [
            self.root / "README.md",
            self.root / "docs" / "index.md",
            self.root / "docs" / "setup" / "installation.md",
        ]
        missing = [str(path) for path in required if not path.exists()]
        return len(missing) == 0, "missing: none" if not missing else ",".join(missing)

    def _precommit_working(self) -> tuple[bool, str]:
        hooks = self.root / ".git" / "hooks"
        ok = (
            (hooks / "pre-commit").exists()
            and (hooks / "commit-msg").exists()
            and (hooks / "pre-push").exists()
        )
        return ok, "hooks present" if ok else "hooks missing"

    def _end_to_end(self) -> tuple[bool, str]:
        result = run_cmd(
            [
                sys.executable,
                "integrate_track_with_environment.py",
                "--track-id",
                "yas_marina",
                "--episodes",
                "1",
            ],
            cwd=self.root,
        )
        return result.returncode == 0, (result.stdout or result.stderr)[-250:]

    def _perf_benchmark(self) -> tuple[bool, str]:
        t0 = time.perf_counter()
        result = run_cmd(
            [
                sys.executable,
                "scripts/train.py",
                "--config",
                "configs/training_config.json",
            ],
            cwd=self.root,
        )
        dt = time.perf_counter() - t0
        ok = result.returncode == 0 and dt < 20.0
        return ok, f"train-entrypoint-seconds={dt:.2f}"

    def _resource_usage(self) -> tuple[bool, str]:
        import psutil

        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        ok = mem.percent < 95.0
        return ok, f"cpu={cpu:.1f}% ram={mem.percent:.1f}%"

    def _pytest_smoke(self) -> tuple[bool, str]:
        result = run_cmd(
            [sys.executable, "-m", "pytest", "-q", "tests/test_environment.py"],
            cwd=self.root,
        )
        return result.returncode == 0, (result.stdout or result.stderr)[-250:]

    def _phase_marker(self) -> tuple[bool, str]:
        required_scripts = [
            "scripts/prepare_initial_commit.py",
            "scripts/create_initial_commit.py",
            "scripts/setup_remote_repository.py",
            "scripts/install_precommit_hooks.py",
        ]
        missing = [path for path in required_scripts if not (self.root / path).exists()]
        return not missing, "missing none" if not missing else ",".join(missing)

    def _ci_workflows(self) -> tuple[bool, str]:
        wf = self.root / ".github" / "workflows"
        files = sorted(path.name for path in wf.glob("*.yml"))
        return len(files) >= 3, ",".join(files)

    def _requirements_lock(self) -> tuple[bool, str]:
        req = self.root / "requirements.txt"
        dev = self.root / "requirements-dev.txt"
        ok = (
            req.exists()
            and dev.exists()
            and req.stat().st_size > 0
            and dev.stat().st_size > 0
        )
        return (
            ok,
            f"requirements={human_size(req.stat().st_size)} dev={human_size(dev.stat().st_size)}",
        )

    def _git_clean_state(self) -> tuple[bool, str]:
        status = run_cmd(["git", "status", "--porcelain"], cwd=self.root)
        clean = len(status.stdout.strip()) == 0
        return clean, (
            "clean" if clean else f"dirty lines={len(status.stdout.splitlines())}"
        )

    def _report_dirs(self) -> tuple[bool, str]:
        out = self.root / "docs" / "reports"
        out.mkdir(parents=True, exist_ok=True)
        return out.exists(), str(out)

    def _execute(self) -> None:
        tests: list[tuple[int, str, bool, Callable[[], tuple[bool, str]], str]] = [
            (1, "Python version check", True, self._python_version, "environment"),
            (2, "Virtual environment active", True, self._venv_active, "environment"),
            (
                3,
                "Dependency conflict check",
                True,
                self._requirements_installed,
                "environment",
            ),
            (4, "Assetto Corsa detection", True, self._assetto_detect, "environment"),
            (
                5,
                "Gym environment creation",
                True,
                self._gym_env_creation,
                "environment",
            ),
            (
                6,
                "Random agent episode smoke",
                True,
                self._random_episode,
                "environment",
            ),
            (7, "Yas Marina installation", True, self._yas_install, "track"),
            (8, "Track data extraction integrity", True, self._track_data, "track"),
            (
                9,
                "Track visualization generation",
                False,
                self._track_visualization,
                "track",
            ),
            (10, "Track integration", True, self._track_integration, "track"),
            (11, "CUDA availability", True, self._cuda_available, "gpu"),
            (12, "GPU detection", True, self._gpu_detect, "gpu"),
            (13, "PyTorch GPU ops", True, self._pytorch_gpu_ops, "gpu"),
            (14, "Mixed precision", True, self._mixed_precision, "gpu"),
            (15, "GPU memory management", True, self._memory_management, "gpu"),
            (16, "Batch size optimization", False, self._batch_opt, "gpu"),
            (17, "Git repository health", True, self._git_repo, "repo"),
            (18, "Project structure validation", True, self._project_structure, "repo"),
            (19, "Configuration file validation", True, self._config_files, "repo"),
            (20, "Module import tests", True, self._import_tests, "repo"),
            (21, "Documentation presence", False, self._documentation, "repo"),
            (22, "Pre-commit hooks presence", False, self._precommit_working, "repo"),
            (
                23,
                "End-to-end integration workflow",
                True,
                self._end_to_end,
                "integration",
            ),
            (24, "Performance benchmark", False, self._perf_benchmark, "integration"),
            (25, "Resource usage sanity", False, self._resource_usage, "integration"),
            (26, "Pytest smoke run", False, self._pytest_smoke, "integration"),
            (27, "Bootstrap scripts presence", False, self._phase_marker, "repo"),
            (28, "CI workflow presence", False, self._ci_workflows, "repo"),
            (29, "Requirements baseline", False, self._requirements_lock, "repo"),
            (30, "Git clean state", False, self._git_clean_state, "repo"),
            (31, "Report directory ready", False, self._report_dirs, "repo"),
        ]

        # Preserve requested 25 tests by running first 25 checks only.
        selected = tests[:25]
        for idx, name, critical, fn, group in selected:
            if self.subset != "all" and self.subset != group:
                continue
            self.run_case(idx, name, critical, fn)

    def report(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        passed = sum(1 for test in self.results if test.passed)
        failed = len(self.results) - passed
        critical_failed = [
            test for test in self.results if (not test.passed and test.critical)
        ]

        payload = {
            "timestamp": utc_now(),
            "subset": self.subset,
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "critical_failed": len(critical_failed),
            },
            "results": [asdict(test) for test in self.results],
        }

        write_json(output_dir / "phase1_test_report.json", payload)

        md_lines = [
            "# Phase 1 Test Report",
            "",
            f"- Total: {len(self.results)}",
            f"- Passed: {passed}",
            f"- Failed: {failed}",
            f"- Critical failed: {len(critical_failed)}",
            "",
            "## Results",
        ]
        for test in self.results:
            status = "PASS" if test.passed else "FAIL"
            md_lines.append(
                f"- [{status}] T{test.id}: {test.name} ({test.duration_ms:.1f} ms) - {test.message}"
            )
        write_text(output_dir / "phase1_test_report.md", "\n".join(md_lines) + "\n")

        html = [
            "<html><head><title>Phase 1 Test Report</title></head><body>",
            "<h1>Phase 1 Test Report</h1>",
            f"<p>Total: {len(self.results)} | Passed: {passed} | Failed: {failed}</p>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr><th>ID</th><th>Name</th><th>Status</th><th>Critical</th><th>Duration(ms)</th><th>Message</th></tr>",
        ]
        for test in self.results:
            status = "PASS" if test.passed else "FAIL"
            html.append(
                "<tr>"
                f"<td>{test.id}</td><td>{test.name}</td><td>{status}</td>"
                f"<td>{test.critical}</td><td>{test.duration_ms:.1f}</td><td>{test.message}</td>"
                "</tr>"
            )
        html.append("</table></body></html>")
        write_text(output_dir / "phase1_test_report.html", "\n".join(html))

        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comprehensive Phase 1 test suite")
    parser.add_argument(
        "--subset",
        choices=["all", "environment", "track", "gpu", "repo", "integration"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("logs/bootstrap"))
    args = parser.parse_args()

    runner = Phase1TestRunner(subset=args.subset)
    runner._execute()
    payload = runner.report(args.output_dir)

    critical_failed = payload["summary"]["critical_failed"]
    failed = payload["summary"]["failed"]
    print(json.dumps(payload["summary"], indent=2))
    if critical_failed > 0:
        return 2
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
