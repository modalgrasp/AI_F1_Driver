#!/usr/bin/env python3
"""Real-time GPU monitoring dashboard and metric logger."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

pynvml = None
try:
    pynvml = importlib.import_module("pynvml")
except Exception:  # pragma: no cover
    pynvml = None

from rich.console import Console
from rich.live import Live
from rich.table import Table

from utils.logger_config import setup_logging

LOGGER = logging.getLogger(__name__)


class GPUMonitor:
    def __init__(self, poll_interval: float = 1.0, alert_temp_c: int = 80, alert_mem_pct: int = 90) -> None:
        self.poll_interval = poll_interval
        self.alert_temp_c = alert_temp_c
        self.alert_mem_pct = alert_mem_pct
        self.console = Console()

        self._nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False

    def _read_once(self) -> dict[str, Any]:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "cpu_percent": cpu,
            "ram_percent": mem,
            "gpu_util_percent": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
            "temperature_c": 0.0,
            "power_w": 0.0,
            "clock_mhz": 0.0,
            "fan_percent": 0.0,
        }

        if self._nvml_ready:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            try:
                fan = float(pynvml.nvmlDeviceGetFanSpeed(handle))
            except Exception:
                fan = 0.0

            data.update(
                {
                    "gpu_util_percent": float(util.gpu),
                    "gpu_mem_used_mb": round(mem_info.used / (1024**2), 2),
                    "gpu_mem_total_mb": round(mem_info.total / (1024**2), 2),
                    "temperature_c": float(temp),
                    "power_w": round(power, 2),
                    "clock_mhz": float(clock),
                    "fan_percent": fan,
                }
            )

        data["gpu_mem_percent"] = (
            0.0 if data["gpu_mem_total_mb"] <= 0 else (100.0 * data["gpu_mem_used_mb"] / data["gpu_mem_total_mb"])
        )
        return data

    def _alerts(self, sample: dict[str, Any]) -> list[str]:
        alerts: list[str] = []
        if sample["temperature_c"] >= self.alert_temp_c:
            alerts.append("High GPU temperature")
        if sample["gpu_mem_percent"] >= self.alert_mem_pct:
            alerts.append("High GPU memory usage")
        return alerts

    def _table(self, sample: dict[str, Any], alerts: list[str]) -> Table:
        table = Table(title="GPU Monitor")
        table.add_column("Metric")
        table.add_column("Value")
        for key in [
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "gpu_mem_percent",
            "temperature_c",
            "power_w",
            "clock_mhz",
            "fan_percent",
            "cpu_percent",
            "ram_percent",
        ]:
            table.add_row(key, str(round(float(sample[key]), 2)))
        table.add_row("alerts", ", ".join(alerts) if alerts else "none")
        return table

    def run(self, duration_sec: int = 30, csv_path: str | Path = "logs/gpu_metrics.csv") -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "timestamp",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "gpu_mem_percent",
            "temperature_c",
            "power_w",
            "clock_mhz",
            "fan_percent",
            "cpu_percent",
            "ram_percent",
        ]

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()

            with Live(console=self.console, refresh_per_second=4) as live:
                end = time.time() + duration_sec
                while time.time() < end:
                    sample = self._read_once()
                    alerts = self._alerts(sample)
                    if alerts:
                        LOGGER.warning("; ".join(alerts))

                    writer.writerow({field: sample[field] for field in fields})
                    live.update(self._table(sample, alerts))
                    time.sleep(self.poll_interval)

        summary_path = out.with_suffix(".json")
        summary_path.write_text(json.dumps({"csv": str(out), "duration_sec": duration_sec}, indent=2), encoding="utf-8")
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-time GPU monitor")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    setup_logging(Path("logs"), level="INFO", console=True)
    mon = GPUMonitor(poll_interval=args.poll)
    out = mon.run(duration_sec=args.duration)
    print(f"Metrics written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
