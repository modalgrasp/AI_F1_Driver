#!/usr/bin/env python3
"""Experiment tracking and lifecycle management."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetadata:
    experiment_id: str
    created_at: str
    git_commit: str
    seed: int
    hardware: dict[str, Any]


class ExperimentManager:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.experiments = self.root / "experiments"
        self.template = self.experiments / "experiment_template"

    def _git_commit(self) -> str:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    def create_experiment(self, seed: int = 42) -> Path:
        exp_id = f"experiment_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        target = self.experiments / exp_id
        if self.template.exists():
            shutil.copytree(self.template, target)
        else:
            (target / "checkpoints").mkdir(parents=True, exist_ok=True)
            (target / "logs").mkdir(parents=True, exist_ok=True)
            (target / "results").mkdir(parents=True, exist_ok=True)

        meta = ExperimentMetadata(
            experiment_id=exp_id,
            created_at=datetime.now(UTC).isoformat(),
            git_commit=self._git_commit(),
            seed=seed,
            hardware={},
        )
        (target / "metadata.json").write_text(
            json.dumps(asdict(meta), indent=2), encoding="utf-8"
        )
        (target / "README.md").write_text(
            "# Experiment\n\nTODO: describe hypothesis and settings.\n",
            encoding="utf-8",
        )
        return target

    def log_metrics(self, experiment_path: Path, metrics: dict[str, Any]) -> None:
        metrics_file = experiment_path / "results" / "metrics.jsonl"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"timestamp": datetime.now(UTC).isoformat(), **metrics})
                + "\n"
            )

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for exp in experiment_ids:
            path = self.experiments / exp / "results" / "metrics.jsonl"
            if not path.exists():
                continue
            lines = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                rows.append({"experiment": exp, "latest": lines[-1]})
        return rows

    def resume_info(self, experiment_id: str) -> dict[str, Any]:
        base = self.experiments / experiment_id
        checkpoints = sorted((base / "checkpoints").glob("*"))
        latest = str(checkpoints[-1]) if checkpoints else ""
        return {"experiment": experiment_id, "latest_checkpoint": latest}
