#!/usr/bin/env python3
"""GPU profiling utilities for development and production."""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import torch

LOGGER = logging.getLogger(__name__)


def profile_gpu_function(name: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            LOGGER.info("[GPU-PROFILE] %s took %.6fs", label, dt)
            return result

        return wrapper

    return decorator


class GPUProfiler:
    def __init__(self, log_dir: str | Path = "logs/profiler") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def profile_callable(
        self, fn: Callable[[], Any], trace_name: str = "profile_run"
    ) -> dict[str, Any]:
        trace_path = self.log_dir / f"{trace_name}.json"
        if not torch.cuda.is_available():
            t0 = time.perf_counter()
            fn()
            return {"cuda": False, "seconds": time.perf_counter() - t0, "trace": None}

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.log_dir)),
        ) as prof:
            fn()
            prof.step()

        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        summary_path = self.log_dir / f"{trace_name}_summary.txt"
        summary_path.write_text(table, encoding="utf-8")

        chrome = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=50
        )
        trace_path.write_text(json.dumps({"summary": chrome}), encoding="utf-8")

        return {"cuda": True, "summary": str(summary_path), "trace": str(trace_path)}

    def quick_memory_timeline(
        self, output: str | Path = "logs/profiler/memory_timeline.json"
    ) -> Path:
        out = Path(output)
        if not torch.cuda.is_available():
            out.write_text(json.dumps({"cuda": False}, indent=2), encoding="utf-8")
            return out

        points = []
        for _ in range(10):
            points.append(
                {
                    "allocated": torch.cuda.memory_allocated(0),
                    "reserved": torch.cuda.memory_reserved(0),
                    "max_allocated": torch.cuda.max_memory_allocated(0),
                }
            )
            time.sleep(0.1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(points, indent=2), encoding="utf-8")
        return out
