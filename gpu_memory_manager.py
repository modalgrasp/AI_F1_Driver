#!/usr/bin/env python3
"""Dynamic GPU memory manager and OOM-safe utilities."""

from __future__ import annotations

import contextlib
import gc
import logging
from dataclasses import dataclass
from typing import Generator

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryReport:
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float


class GPUMemoryManager:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._tracked_tensors: list[torch.Tensor] = []

    def available(self) -> bool:
        return torch.cuda.is_available()

    def report(self) -> MemoryReport:
        if not self.available():
            return MemoryReport(0.0, 0.0, 0.0)
        idx = int(self.device.split(":")[-1])
        return MemoryReport(
            allocated_gb=round(torch.cuda.memory_allocated(idx) / (1024**3), 4),
            reserved_gb=round(torch.cuda.memory_reserved(idx) / (1024**3), 4),
            peak_allocated_gb=round(
                torch.cuda.max_memory_allocated(idx) / (1024**3), 4
            ),
        )

    def track(self, tensor: torch.Tensor) -> torch.Tensor:
        self._tracked_tensors.append(tensor)
        return tensor

    def clear_tracked(self) -> None:
        self._tracked_tensors.clear()
        gc.collect()
        if self.available():
            torch.cuda.empty_cache()

    def preallocation_check(self, required_gb: float) -> bool:
        if not self.available():
            return False
        idx = int(self.device.split(":")[-1])
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        current = torch.cuda.memory_reserved(idx) / (1024**3)
        return current + required_gb <= total

    def safe_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        required_gb = torch.tensor([], dtype=dtype).element_size()
        required_gb = (required_gb * int(torch.tensor(shape).prod().item())) / (1024**3)
        if not self.preallocation_check(required_gb * 1.2):
            raise RuntimeError(
                "OOM prevention triggered: requested tensor exceeds safe budget"
            )
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        return self.track(tensor)

    @contextlib.contextmanager
    def temporary_allocation(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Generator[torch.Tensor, None, None]:
        tensor = self.safe_tensor(shape, dtype=dtype)
        try:
            yield tensor
        finally:
            del tensor
            gc.collect()
            if self.available():
                torch.cuda.empty_cache()

    def auto_tune_batch_size(
        self, trial_fn, low: int = 8, high: int = 4096, target_util: float = 0.88
    ) -> int:
        """Binary search highest stable batch size using provided trial callback.

        trial_fn must accept batch_size and run one mini training step.
        """
        best = low
        while low <= high:
            mid = (low + high) // 2
            try:
                trial_fn(mid)
                if self.available():
                    idx = int(self.device.split(":")[-1])
                    util = (
                        torch.cuda.memory_reserved(idx)
                        / torch.cuda.get_device_properties(idx).total_memory
                    )
                else:
                    util = 0.0
                if util <= target_util:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if self.available():
                        torch.cuda.empty_cache()
                    high = mid - 1
                else:
                    raise
        return best

    def suggest_gradient_accumulation(
        self, desired_effective_batch: int, max_batch: int
    ) -> int:
        if max_batch <= 0:
            return 1
        return max(1, (desired_effective_batch + max_batch - 1) // max_batch)

    def fragmentation_indicator(self) -> float:
        if not self.available():
            return 0.0
        rep = self.report()
        if rep.reserved_gb <= 0:
            return 0.0
        return round(
            max(0.0, (rep.reserved_gb - rep.allocated_gb) / rep.reserved_gb), 4
        )

    def defragment_if_needed(self, threshold: float = 0.35) -> bool:
        frag = self.fragmentation_indicator()
        if frag >= threshold:
            self.clear_tracked()
            LOGGER.warning("Defragmentation triggered at indicator %.3f", frag)
            return True
        return False
