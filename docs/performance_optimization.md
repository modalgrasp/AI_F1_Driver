# Performance Optimization Guide

## Overview
Vehicle dynamics simulation throughput is critical for RL training. This guide covers profiling, benchmarks, and optimization hooks added in Phase 2.6.

## Performance Targets
- Single step: < 1 ms (1000+ steps/second)
- Parallel (8 envs): > 5000 total steps/second
- Memory per vehicle: < 100 MB
- Memory growth over long runs: near-zero

## Optimization Techniques Applied

### 1. Vectorization
- Added batched state packing and batch update support in optimization utilities.
- Reduced per-step Python overhead for RL rollouts by exposing compact NumPy state vectors.

### 2. Caching
- Added cached lookup wrappers for aerodynamic coefficients.
- Avoids repeated recomputation for stable aero modes.

### 3. JIT Compilation
- Added optional Numba micro-kernel path in optimization utilities.
- Falls back safely when Numba is unavailable.

### 4. Memory Layout
- Added contiguous packed vehicle state representation for replay buffers.
- Helps cache locality and reduces conversion overhead.

## Profiling
```python
from optimization.profiler import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.profile_single_step()
profiler.profile_subsystems()
profiler.profile_memory_usage()
profiler.generate_report()
```

## Benchmarks
```bash
python optimization/benchmarks.py
```

This writes benchmark artifacts to optimization/benchmark_results.json.

## CI Performance Gates
- tests/performance/test_performance_regression.py enforces throughput and memory-growth constraints.
- Thresholds are configurable by environment variables:
  - F1_MIN_STEPS_PER_SEC
  - F1_MIN_PARALLEL_STEPS_PER_SEC
  - F1_MAX_MEMORY_GROWTH_BYTES

## RL Integration Notes
- Use optimization.optimizations.VehicleOptimizations.batch_computations for multi-env stepping.
- Store packed states directly in replay buffers to reduce Python object overhead.
- Run profiler and benchmarks after any vehicle_dynamics changes.
