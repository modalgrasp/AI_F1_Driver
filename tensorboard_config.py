#!/usr/bin/env python3
"""TensorBoard logging helpers with fallback when tensorboard is unavailable."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


class TensorboardLogger:
    def __init__(self, log_dir: str | Path = "logs/tensorboard") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = (
            SummaryWriter(str(self.log_dir)) if SummaryWriter is not None else None
        )
        self.fallback_file = self.log_dir / "tensorboard_fallback.jsonl"

    def _fallback_write(self, payload: dict[str, Any]) -> None:
        with self.fallback_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def log_train_metrics(
        self, step: int, loss: float, reward: float, episode_len: int, lr: float
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "loss": loss,
            "reward": reward,
            "episode_len": episode_len,
            "lr": lr,
        }
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/reward", reward, step)
            self.writer.add_scalar("train/episode_len", episode_len, step)
            self.writer.add_scalar("train/lr", lr, step)
        else:
            self._fallback_write({"type": "train", **payload})

    def log_gpu_metrics(
        self, step: int, gpu_util: float, vram_gb: float, temp_c: float
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "gpu_util": gpu_util,
            "vram_gb": vram_gb,
            "temp_c": temp_c,
        }
        if self.writer is not None:
            self.writer.add_scalar("gpu/util", gpu_util, step)
            self.writer.add_scalar("gpu/vram_gb", vram_gb, step)
            self.writer.add_scalar("gpu/temp_c", temp_c, step)
        else:
            self._fallback_write({"type": "gpu", **payload})

    def log_hparams(self, params: dict[str, Any], metrics: dict[str, float]) -> None:
        if self.writer is not None:
            self.writer.add_hparams(params, metrics)
        else:
            self._fallback_write(
                {"type": "hparams", "params": params, "metrics": metrics}
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
