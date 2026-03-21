"""Common pytest fixtures for F1 project."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_workdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_track_data() -> dict:
    return {
        "track_id": "yas_marina",
        "length_km": 5.281,
        "turns": 16,
    }


@pytest.fixture
def dummy_env_config(tmp_path: Path) -> Path:
    cfg = {
        "assetto_corsa": {"install_path": "D:/SteamLibrary/steamapps/common/assettocorsa"},
        "training": {"max_episode_steps": 1000},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def mock_gpu(monkeypatch):
    class _MockCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    monkeypatch.setattr("torch.cuda", _MockCuda)
