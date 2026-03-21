# F1 Autonomous Racing AI Setup Guide (Phase 1 - Step 1.1)

## 1. Purpose
This guide installs and validates a production-ready baseline for integrating Assetto Corsa telemetry with a Gymnasium RL environment for Yas Marina autonomous racing experiments.

## 2. Prerequisites
- Windows 11 (recommended for Assetto Corsa integration)
- Python 3.10+
- NVIDIA GPU drivers with CUDA 12.1+ compatibility
- Assetto Corsa installed through Steam

## 3. Installation
1. Open PowerShell in project root.
2. Run:
   ```bat
   setup.bat
   ```
3. Wait for dependency installation and CUDA validation.
4. Run full validation:
   ```bat
   f1_racing_env\Scripts\activate
   python validate_setup.py
   ```

## 4. Configure Assetto Corsa Path
Edit `configs/config.json`:
- Set `assetto_corsa.install_path` to your local install path if auto-detection fails.
- Keep `race.track` as `yas_marina`.

## 5. Verify Environment
Run environment smoke test:
```bat
python tests/test_environment.py
```

Expected behavior:
- 5 random-policy episodes execute
- Logs appear under `logs/`
- Average step duration and memory growth statistics are printed

## 6. Troubleshooting
### Assetto Corsa not found
- Confirm Steam install path exists
- Set explicit path in `configs/config.json`
- Re-run `python validate_setup.py`

### CUDA not available
- Check `nvidia-smi` output
- Reinstall PyTorch CUDA wheel from requirements
- Confirm NVIDIA driver supports CUDA 12.x

### Shared memory access failure
- Launch Assetto Corsa and enter a session first
- Ensure telemetry/shared memory plugins are enabled
- Re-run validation script

### Import errors
- Activate environment before running scripts
- Re-run `pip install -r requirements.txt`

## 7. Next Step (Phase 2 Preview)
- Replace action dispatch stub with real game control bridge
- Expand shared memory parsing to full Assetto Corsa structs
- Integrate PPO/SAC training loop with checkpointing
