#!/usr/bin/env python3
"""Auto-discover batch size sweet spot for GPU training throughput."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def training_trial(
    batch_size: int, feature_dim: int = 512, hidden: int = 1024
) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = "cuda:0"
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 256),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn((batch_size, feature_dim), device=device)
    y = torch.randn((batch_size, 256), device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    samples_per_sec = (batch_size * 10) / max(dt, 1e-9)
    mem_pct = (
        torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory
    )
    return samples_per_sec, mem_pct


def optimize(
    min_bs: int, max_bs: int, target_low: float, target_high: float
) -> dict[str, Any]:
    best = min_bs
    best_samples = 0.0
    fallback_best = min_bs
    fallback_samples = 0.0
    lo, hi = min_bs, max_bs

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            sps, mem = training_trial(mid)
            if sps > fallback_samples:
                fallback_samples = sps
                fallback_best = mid
            if target_low <= mem <= target_high and sps > best_samples:
                best = mid
                best_samples = sps
            if mem < target_low:
                lo = mid + 1
            elif mem > target_high:
                hi = mid - 1
            else:
                lo = mid + 1
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                hi = mid - 1
            else:
                raise

    # Parallel environment recommendation from CPU core count and GPU reserve.
    if best_samples <= 0.0:
        best = fallback_best
        best_samples = fallback_samples

    cpu_cores = max(1, (torch.get_num_threads() or 8))
    envs = min(16, max(2, cpu_cores // 2))
    return {
        "optimal_batch_size": best,
        "samples_per_sec": best_samples,
        "recommended_parallel_envs": envs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch size optimizer for CUDA training"
    )
    parser.add_argument("--min-batch", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument("--target-low", type=float, default=0.80)
    parser.add_argument("--target-high", type=float, default=0.95)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/batch_size_recommendation.json")
    )
    args = parser.parse_args()

    result = optimize(args.min_batch, args.max_batch, args.target_low, args.target_high)
    payload = {"timestamp": datetime.now(UTC).isoformat(), **result}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
