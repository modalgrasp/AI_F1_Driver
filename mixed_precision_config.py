#!/usr/bin/env python3
"""Mixed precision training configuration and wrappers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class AMPMetrics:
    steps: int
    seconds: float
    iter_per_sec: float
    peak_memory_gb: float


class MixedPrecisionManager:
    def __init__(self, enabled: bool = True, dtype: str = "float16", init_scale: float = 65536.0) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.autocast_dtype = torch.float16 if dtype.lower() == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled, init_scale=init_scale)

    def autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.enabled)

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

    def train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
    ) -> float:
        with self.autocast():
            outputs = model(inputs)
            # Keep some numerically sensitive reductions in FP32 if needed.
            loss = loss_fn(outputs.float(), targets.float())
        self.backward_step(loss, optimizer)
        return float(loss.detach().item())

    def benchmark_wrapper(self, model: torch.nn.Module, batch_shape: tuple[int, ...], steps: int = 100) -> AMPMetrics:
        if not torch.cuda.is_available():
            return AMPMetrics(steps=0, seconds=0.0, iter_per_sec=0.0, peak_memory_gb=0.0)
        device = "cuda:0"
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        x = torch.randn(batch_shape, device=device)
        y = torch.randn((batch_shape[0], model(x).shape[-1]), device=device)

        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = self.train_step(model, optimizer, x, y, loss_fn)
        torch.cuda.synchronize(0)
        dt = time.perf_counter() - t0

        peak = torch.cuda.max_memory_allocated(0) / (1024**3)
        return AMPMetrics(steps=steps, seconds=dt, iter_per_sec=steps / max(dt, 1e-9), peak_memory_gb=peak)


def rl_amp_example_step(policy: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    amp = MixedPrecisionManager(enabled=True, dtype="float16")
    obs = batch["obs"].to("cuda")
    target = batch["target"].to("cuda")
    loss_fn = torch.nn.MSELoss()
    loss = amp.train_step(policy, optimizer, obs, target, loss_fn)
    return {"loss": loss, "scale": float(amp.scaler.get_scale())}
