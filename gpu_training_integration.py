#!/usr/bin/env python3
"""GPU integration helpers for RL training presets and data path optimization."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PPO = None
try:
    _sb3 = importlib.import_module("stable_baselines3")
    PPO = getattr(_sb3, "PPO", None)
except Exception:  # pragma: no cover
    PPO = None

from gpu_config_manager import GPUConfigManager


@dataclass
class TrainingPreset:
    name: str
    batch_size: int
    learning_rate: float
    n_steps: int
    use_amp: bool
    gradient_accumulation: int


PRESETS = {
    "maximum_speed": TrainingPreset("maximum_speed", batch_size=2048, learning_rate=3e-4, n_steps=4096, use_amp=True, gradient_accumulation=1),
    "balanced": TrainingPreset("balanced", batch_size=1024, learning_rate=3e-4, n_steps=2048, use_amp=True, gradient_accumulation=1),
    "memory_efficient": TrainingPreset("memory_efficient", batch_size=512, learning_rate=2e-4, n_steps=1024, use_amp=True, gradient_accumulation=2),
}


class GPUReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cuda:0") -> None:
        self.capacity = capacity
        self.device = device if torch.cuda.is_available() else "cpu"
        self.obs = torch.empty((capacity, obs_dim), device=self.device)
        self.next_obs = torch.empty((capacity, obs_dim), device=self.device)
        self.actions = torch.empty((capacity, action_dim), device=self.device)
        self.rewards = torch.empty((capacity, 1), device=self.device)
        self.dones = torch.empty((capacity, 1), device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


def configure_dataloader_like_settings() -> dict[str, Any]:
    return {
        "pin_memory": True,
        "num_workers": 4,
        "prefetch_factor": 4,
    }


def configure_sb3_device(preset: TrainingPreset, env_id: str = "CartPole-v1") -> dict[str, Any]:
    if PPO is None:
        return {"available": False, "reason": "stable_baselines3 not installed for this Python version"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("MlpPolicy", env_id, device=device, batch_size=preset.batch_size, n_steps=preset.n_steps, learning_rate=preset.learning_rate, verbose=0)
    return {"available": True, "device": device, "policy": str(type(model.policy).__name__)}


def dummy_training_loop(preset: TrainingPreset, steps: int = 200) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=preset.learning_rate)

    amp_enabled = preset.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for _ in range(steps):
        x = torch.randn((preset.batch_size, 128), device=device)
        y = torch.randn((preset.batch_size, 64), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            out = model(x)
            loss = torch.nn.functional.mse_loss(out.float(), y.float())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"final_loss": float(loss.detach().item()), "device": device}


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU training integration presets")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced")
    parser.add_argument("--output", type=Path, default=Path("logs/gpu_training_integration_report.json"))
    args = parser.parse_args()

    mgr = GPUConfigManager("configs/gpu_config.json")
    mgr.configure_torch()

    preset = PRESETS[args.preset]
    sb3_info = configure_sb3_device(preset)
    dummy = dummy_training_loop(preset)
    data_cfg = configure_dataloader_like_settings()

    payload = {
        "preset": preset.__dict__,
        "allocation_strategy": mgr.allocation_strategy(),
        "sb3": sb3_info,
        "dummy_training": dummy,
        "data_pipeline": data_cfg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
