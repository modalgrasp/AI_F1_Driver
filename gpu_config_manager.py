#!/usr/bin/env python3
"""GPU configuration manager for concurrent simulation + training."""

from __future__ import annotations

import gc
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    capability: str
    total_vram_gb: float
    multiprocessor_count: int
    driver: str | None
    cuda_runtime: str | None


class GPUConfigManager:
    def __init__(self, config_path: str | Path = "configs/gpu_config.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_or_default()

    def _load_or_default(self) -> dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        default = {
            "device": "cuda:0",
            "simulation_vram_reserve_gb": 2.5,
            "training_vram_target_gb": 7.0,
            "system_buffer_gb": 2.0,
            "torch": {
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "allow_tf32": True,
                "matmul_precision": "high",
            },
        }
        self.save(default)
        return default

    def save(self, config: dict[str, Any] | None = None) -> None:
        payload = config or self.config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def detect_gpus(self) -> list[GPUInfo]:
        driver = None
        runtime = None
        try:
            out = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            ).stdout
            for line in out.splitlines():
                if "Driver Version:" in line and "CUDA Version:" in line:
                    driver = line.split("Driver Version:", 1)[1].split()[0]
                    runtime = line.split("CUDA Version:", 1)[1].split()[0]
                    break
        except OSError:
            pass

        infos: list[GPUInfo] = []
        if not torch.cuda.is_available():
            return infos

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cap = torch.cuda.get_device_capability(i)
            infos.append(
                GPUInfo(
                    index=i,
                    name=torch.cuda.get_device_name(i),
                    capability=f"{cap[0]}.{cap[1]}",
                    total_vram_gb=round(props.total_memory / (1024**3), 2),
                    multiprocessor_count=props.multi_processor_count,
                    driver=driver,
                    cuda_runtime=runtime,
                )
            )
        return infos

    def configure_torch(self) -> None:
        cfg = self.config.get("torch", {})
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = bool(cfg.get("cudnn_deterministic", False))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.set_float32_matmul_precision(str(cfg.get("matmul_precision", "high")))

    def allocation_strategy(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "simulation_gb": 0.0,
                "training_gb": 0.0,
                "buffer_gb": 0.0,
                "total_gb": 0.0,
            }

        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        sim = float(self.config.get("simulation_vram_reserve_gb", 2.5))
        buffer = float(self.config.get("system_buffer_gb", 2.0))
        train = min(
            float(self.config.get("training_vram_target_gb", 7.0)),
            max(0.0, total - sim - buffer),
        )
        return {
            "simulation_gb": round(sim, 2),
            "training_gb": round(train, 2),
            "buffer_gb": round(buffer, 2),
            "total_gb": round(total, 2),
        }

    def memory_stats(self, device: str = "cuda:0") -> dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
        idx = int(device.split(":")[-1])
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved(idx) / (1024**3), 3),
            "max_allocated_gb": round(
                torch.cuda.max_memory_allocated(idx) / (1024**3), 3
            ),
        }

    def clear_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def leak_snapshot(self, device: str = "cuda:0") -> dict[str, float]:
        return self.memory_stats(device)

    def compare_snapshots(
        self, before: dict[str, float], after: dict[str, float]
    ) -> dict[str, float]:
        return {
            key: round(after.get(key, 0.0) - before.get(key, 0.0), 4) for key in before
        }

    def export_inventory(self, output: str | Path = "logs/gpu_inventory.json") -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gpus": [asdict(item) for item in self.detect_gpus()],
            "allocation": self.allocation_strategy(),
            "config": self.config,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
