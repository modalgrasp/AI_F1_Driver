#!/usr/bin/env python3
"""Main runner for complete model validation and calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validation.calibration import ParameterCalibrator
from validation.report_generator import ValidationReportGenerator
from validation.validation_framework import ReferenceData, ValidationFramework
from validation.validation_plots import (
    plot_acceleration_curve,
    plot_aero_validation,
    plot_tire_validation_curves,
)
from vehicle_dynamics.vehicle_model import VehicleDynamicsModel


def run_complete_validation(
    run_calibration: bool = True, fast_mode: bool = False
) -> dict[str, Any]:
    """Run all validators, optional calibration, and report generation."""
    vehicle = VehicleDynamicsModel()
    ref = ReferenceData.build_default()

    print("Running validation tests...")
    framework = ValidationFramework(
        vehicle_model=vehicle, reference_data=ref, fast_mode=fast_mode
    )
    validation_bundle = framework.run_complete_validation(include_lap=not fast_mode)

    if run_calibration:
        print("Running calibration...")
        calibrator = ParameterCalibrator(vehicle_model=vehicle, reference_data=ref)
        calibration = calibrator.calibrate_all()
    else:
        calibration = {}

    full_payload = {
        "validation": validation_bundle,
        "calibration": calibration,
    }

    print("Generating reports...")
    report = ValidationReportGenerator(full_payload["validation"])
    report.generate_markdown_report("docs/validation_report.md")
    report.generate_json_report("validation/validation_results.json")
    report.generate_html_report("docs/validation_report.html")

    Path("validation").mkdir(parents=True, exist_ok=True)
    Path("validation/calibration_results.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    print("Generating plots...")
    plot_tire_validation_curves(vehicle.tire_models[0], output_dir="validation/plots")
    plot_aero_validation(vehicle.aero_model, output_dir="validation/plots")
    plot_acceleration_curve(vehicle, output_dir="validation/plots")

    summary = validation_bundle["summary"]
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"Subsystems passed: {summary['subsystems_passed']} / {summary['subsystems_total']}"
    )
    for subsystem, score in summary["confidence"].items():
        print(f"- {subsystem}: confidence={score:.2f}")
    print("=" * 60)

    return full_payload


if __name__ == "__main__":
    run_complete_validation(run_calibration=True, fast_mode=False)
