# CUDA Setup Guide for RTX 5070 Ti (Step 1.3)

## 1) Check Current Driver and GPU
Windows:
1. Run: nvidia-smi
2. Note Driver Version and CUDA Version shown by driver.
3. Run: nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,power.limit --format=csv

Linux:
1. Run: nvidia-smi
2. Run: nvidia-smi -q | grep -E "Driver Version|CUDA Version|Product Name"

Expected GPU name contains RTX 5070 Ti and VRAM about 12 GB.

## 2) Verify CUDA Compatibility
1. Driver must support CUDA 12.x or 13.x runtime for current PyTorch wheels.
2. Keep NVIDIA driver current from official NVIDIA package only.
3. For this project, prefer CUDA runtime from PyTorch wheels and toolkit for optional custom kernels.

## 3) Install CUDA Toolkit (12.x latest stable)
Windows:
1. Download from NVIDIA CUDA Toolkit page.
2. Install with default components: CUDA compiler, runtime, Nsight tools.
3. Verify:
   - nvcc --version
   - nvidia-smi

Linux:
1. Install through distro-specific package or NVIDIA runfile.
2. Verify:
   - nvcc --version
   - nvidia-smi

## 4) Install cuDNN (matching major CUDA)
Windows:
1. Download cuDNN for matching CUDA major.
2. Copy bin/lib/include files into CUDA Toolkit folders.
3. Verify via PyTorch script in test_gpu_setup.py.

Linux:
1. Install libcudnn package for matching CUDA major.
2. Verify with ldconfig -p | grep cudnn.

## 5) Environment Variables
Windows (PowerShell profile or system env):
1. CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
2. Add to PATH:
   - %CUDA_HOME%\bin
   - %CUDA_HOME%\libnvvp

Linux (bashrc/zshrc):
1. export CUDA_HOME=/usr/local/cuda-12.x
2. export PATH=$CUDA_HOME/bin:$PATH
3. export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## 6) Compute Capability Check
Use:
1. python test_gpu_setup.py --quick
2. or python -c "import torch; print(torch.cuda.get_device_capability(0))"

## 7) Multi-CUDA Version Management
Windows:
1. Keep separate toolkit folders (v12.4, v12.6, etc).
2. Point CUDA_HOME to selected version.

Linux:
1. Use /usr/local/cuda symlink to active version.
2. Update symlink and restart shell.

## 8) Troubleshooting
1. CUDA not detected:
   - Reinstall NVIDIA driver.
   - Ensure you run project venv Python, not system Python.
2. cuDNN init errors:
   - Mismatch between PyTorch CUDA build and installed driver/toolkit.
3. nvcc missing:
   - Toolkit not installed or PATH not set.
4. OOM during training:
   - Use batch_size_optimizer.py and mixed_precision_config.py.
5. Thermal throttling:
   - Reduce power target, improve cooling, cap FPS for simulation.

## 9) Recommended for this project
1. Reserve 2-3 GB VRAM for Assetto Corsa rendering.
2. Use mixed precision for training.
3. Keep temperature under 80 C with monitor alerts.
