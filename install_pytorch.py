#!/usr/bin/env python3
"""Intelligent PyTorch CUDA installer and verifier."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


def run_cmd(command: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out.strip()
    except OSError as exc:
        return 1, str(exc)


def detect_cuda_runtime() -> dict[str, Any]:
    code, out = run_cmd(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"])
    if code != 0:
        return {"available": False, "error": out}

    code2, out2 = run_cmd(["nvidia-smi"])
    cuda_version = None
    for line in out2.splitlines():
        if "CUDA Version:" in line:
            cuda_version = line.split("CUDA Version:", 1)[1].split()[0]
            break

    return {
        "available": True,
        "cuda_driver_runtime": cuda_version,
        "gpus": [row.strip() for row in out.splitlines() if row.strip()],
    }


def recommended_index(cuda_runtime: str | None) -> str:
    if cuda_runtime is None:
        return "https://download.pytorch.org/whl/cu130"
    try:
        major = int(cuda_runtime.split(".")[0])
    except Exception:
        return "https://download.pytorch.org/whl/cu130"
    if major >= 13:
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu121"


def install_torch(index_url: str, dry_run: bool = False) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        index_url,
    ]
    if dry_run:
        return {"status": "dry_run", "command": " ".join(command)}

    code, out = run_cmd(command)
    return {"status": "success" if code == 0 else "failed", "command": " ".join(command), "output": out}


def torch_verify() -> dict[str, Any]:
    script = (
        "import json, torch; "
        "r={'torch_version':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),"
        "'cuda_version':torch.version.cuda,"
        "'device_count':torch.cuda.device_count(),"
        "'cudnn_enabled':bool(torch.backends.cudnn.is_available())};"
        "\n"
        "if torch.cuda.is_available():"
        "\n"
        " d=torch.cuda.get_device_properties(0);"
        " r['device_name']=torch.cuda.get_device_name(0);"
        " r['total_vram_gb']=round(d.total_memory/(1024**3),2);"
        " a=torch.randn((2048,2048),device='cuda');"
        " b=torch.randn((2048,2048),device='cuda');"
        " c=a@b; torch.cuda.synchronize();"
        "\n"
        "print(json.dumps(r))"
    )
    code, out = run_cmd([sys.executable, "-c", script])
    if code != 0:
        return {"status": "failed", "error": out}

    try:
        result = json.loads(out.splitlines()[-1])
    except Exception:
        return {"status": "failed", "error": out}

    result["status"] = "success"
    return result


def tflops_benchmark(matrix_size: int = 4096, repeats: int = 10) -> dict[str, Any]:
    script = (
        "import time, json, torch;"
        f"n={matrix_size}; r={repeats};"
        "ok=torch.cuda.is_available();"
        "res={'available':ok};"
        "\n"
        "if not ok: print(json.dumps(res)); raise SystemExit(0)"
        "\n"
        "a=torch.randn((n,n),device='cuda',dtype=torch.float16);"
        "b=torch.randn((n,n),device='cuda',dtype=torch.float16);"
        "torch.cuda.synchronize(); t0=time.perf_counter();"
        "\n"
        "for _ in range(r): c=torch.matmul(a,b)"
        "\n"
        "torch.cuda.synchronize(); dt=time.perf_counter()-t0;"
        "flops=2*(n**3)*r; tflops=flops/dt/1e12;"
        "res.update({'matrix_size':n,'repeats':r,'seconds':dt,'tflops':tflops});"
        "print(json.dumps(res))"
    )
    code, out = run_cmd([sys.executable, "-c", script])
    if code != 0:
        return {"status": "failed", "error": out}
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        return {"status": "failed", "error": out}
    data["status"] = "success"
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and validate PyTorch CUDA stack")
    parser.add_argument("--index-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback", action="store_true", help="Uninstall torch packages")
    parser.add_argument("--report", type=Path, default=Path("logs/pytorch_install_report.json"))
    args = parser.parse_args()

    setup_logging(Path("logs"), level="INFO", console=True)

    if args.rollback:
        code, out = run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        LOGGER.info("Rollback finished with code %d", code)
        print(out)
        return 0 if code == 0 else 1

    cuda_info = detect_cuda_runtime()
    index_url = args.index_url or recommended_index(cuda_info.get("cuda_driver_runtime"))

    install_result = install_torch(index_url=index_url, dry_run=args.dry_run)
    verify_result = torch_verify() if not args.dry_run else {"status": "skipped"}
    bench_result = tflops_benchmark() if not args.dry_run else {"status": "skipped"}

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "cuda_info": cuda_info,
        "selected_index_url": index_url,
        "install": install_result,
        "verify": verify_result,
        "benchmark": bench_result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Report written to %s", args.report)
    if install_result.get("status") == "failed" or verify_result.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
