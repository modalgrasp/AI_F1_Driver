#!/usr/bin/env python3
"""Tests for validation and calibration framework."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from validation.calibration import ParameterCalibrator
from validation.report_generator import ValidationReportGenerator
from validation.validation_framework import ReferenceData, ValidationFramework
from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


@pytest.mark.unit
def test_reference_data() -> None:
    ref = ReferenceData.build_default()
    assert ref.yas_marina_data["lap_record"] > 0
    assert ref.physics_constraints["max_downforce_coefficient"] < 0


@pytest.mark.integration
def test_validation_framework_runs() -> None:
    vehicle = VehicleDynamicsModel()
    framework = ValidationFramework(vehicle, fast_mode=True)
    payload = framework.run_complete_validation(include_lap=False)

    assert "summary" in payload
    assert "results" in payload
    assert payload["summary"]["subsystems_total"] >= 4
    assert "Tire Model" in payload["results"]
    assert "Aerodynamics Model" in payload["results"]
    assert "Powertrain Model" in payload["results"]
    assert "Vehicle Dynamics" in payload["results"]


@pytest.mark.integration
def test_report_generation(tmp_path: Path) -> None:
    vehicle = VehicleDynamicsModel()
    framework = ValidationFramework(vehicle, fast_mode=True)
    payload = framework.run_complete_validation(include_lap=False)

    gen = ValidationReportGenerator(payload)
    md = tmp_path / "validation.md"
    js = tmp_path / "validation.json"
    html = tmp_path / "validation.html"

    gen.generate_markdown_report(md)
    gen.generate_json_report(js)
    gen.generate_html_report(html)

    assert md.exists()
    assert js.exists()
    assert html.exists()

    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert "summary" in loaded


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_VALIDATION_TESTS") != "1",
    reason="Set RUN_SLOW_VALIDATION_TESTS=1 to run expensive calibration test.",
)
def test_calibrator_runs() -> None:
    vehicle = VehicleDynamicsModel()
    ref = ReferenceData.build_default()
    calibrator = ParameterCalibrator(vehicle, ref)

    tire = calibrator.calibrate_tire_parameters()
    aero = calibrator.calibrate_aero_parameters()
    power = calibrator.calibrate_powertrain_parameters()

    assert tire.success
    assert aero.success
    assert power.success
    assert tire.objective >= 0.0
    assert aero.objective >= 0.0
    assert power.objective >= 0.0
