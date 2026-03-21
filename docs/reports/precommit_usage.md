# Pre-commit Usage Guide

## Run Manually
```bash
pre-commit run --all-files
```

## Emergency Skip (avoid unless absolutely necessary)
```bash
git commit --no-verify
```

## Update Hooks
```bash
pre-commit autoupdate
pre-commit run --all-files
```

## Add New Hook
Edit `.pre-commit-config.yaml`, then run pre-commit install and pre-commit run --all-files.
