#!/usr/bin/env python3
"""Custom CUDA kernels for distance and reward operations.

Builds torch extension when CUDA toolkit is available, otherwise falls back to PyTorch ops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

LOGGER = logging.getLogger(__name__)

_EXT = None


def _sources() -> tuple[str, str]:
    root = Path(__file__).resolve().parent / "cuda_extensions"
    return str(root / "track_ops.cpp"), str(root / "track_ops_kernel.cu")


def build_extension(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT

    cpp, cu = _sources()
    if not Path(cpp).exists() or not Path(cu).exists():
        raise FileNotFoundError("CUDA extension sources not found")

    try:
        _EXT = load(
            name="track_ops_ext",
            sources=[cpp, cu],
            verbose=verbose,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
        )
        return _EXT
    except Exception as exc:
        LOGGER.warning("CUDA extension build failed, using fallback ops: %s", exc)
        _EXT = None
        return None


def pairwise_distance(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
    ext = build_extension(verbose=False)
    if ext is not None and points_a.is_cuda and points_b.is_cuda:
        return ext.pairwise_distance_cuda(points_a, points_b)
    return torch.cdist(points_a, points_b)


def reward_kernel(
    progress: torch.Tensor, offtrack: torch.Tensor, collision: torch.Tensor
) -> torch.Tensor:
    ext = build_extension(verbose=False)
    if ext is not None and progress.is_cuda:
        return ext.reward_kernel_cuda(progress, offtrack, collision)

    return 2.0 * progress - 3.0 * offtrack.float() - 8.0 * collision.float()


def benchmark() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn((4096, 3), device=device)
    b = torch.randn((4096, 3), device=device)
    d = pairwise_distance(a, b)
    if device == "cuda":
        torch.cuda.synchronize()
    return {"device": device, "shape": list(d.shape)}
