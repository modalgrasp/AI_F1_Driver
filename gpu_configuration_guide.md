# GPU Configuration Guide for RTX 5070 Ti 12GB

## Architecture Snapshot
1. GPU: RTX 5070 Ti Laptop
2. VRAM: 12 GB GDDR7
3. CUDA cores: target profile assumes 5888
4. Tensor cores: 5th Gen
5. Recommended target: 80-95% VRAM usage for training while preserving simulation stability

## Compatibility Matrix (Practical)
1. Driver: latest stable NVIDIA Studio/Game Ready supporting CUDA 12/13 runtime
2. PyTorch: CUDA wheel variant matching driver runtime
3. cuDNN: bundled via wheel runtime or matched toolkit installation
4. Python: 3.11/3.12 ideal for full ecosystem compatibility; 3.14 supported with some optional package limits

## Best-Practice PyTorch Runtime Settings
1. torch.backends.cudnn.benchmark=True
2. torch.backends.cudnn.deterministic=False
3. torch.backends.cuda.matmul.allow_tf32=True
4. torch.backends.cudnn.allow_tf32=True
5. torch.set_float32_matmul_precision("high")

## VRAM Budget Strategy
1. Assetto Corsa rendering reserve: 2.5 GB
2. RL model and optimizer: 3.0 GB
3. Replay buffer and tensors: 4.0 GB
4. Safety buffer: 2.5 GB

## Throughput Tuning Flow
1. Run test_gpu_setup.py --quick
2. Run batch_size_optimizer.py
3. Enable mixed precision wrappers
4. Profile with gpu_profiler.py
5. Monitor with gpu_monitor.py and adjust workers/batch size

## Common Issues and Fixes
1. OOM:
   - reduce batch, enable AMP, increase gradient accumulation
2. Low utilization:
   - increase batch, reduce CPU bottlenecks, pin memory
3. Spiky frame times:
   - reserve more VRAM for simulation and cap simulation render settings
4. Slow data transfer:
   - use pinned memory and larger prefetch

## Benchmark Expectations (Guideline)
1. Matrix benchmark TFLOPS result should be stable run-to-run
2. GPU utilization during training should average above 80%
3. Temperature should remain below 80 C for long sessions

## FAQ
1. Can I train and simulate on one GPU?
   - Yes, with VRAM reservation and batch auto-tuning.
2. Should I enable deterministic mode?
   - Only for reproducibility checks; disable for maximum speed.
3. Is TensorBoard required?
   - Optional. Fallback logging is included.
