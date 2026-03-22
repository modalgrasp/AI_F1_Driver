# Validation Methodology

## Overview

This document defines how the custom vehicle dynamics stack is validated and calibrated against real-world F1 behavior. The framework validates component-level and integrated-level dynamics from:

- Tire model
- Aerodynamics model
- Powertrain model
- Full vehicle dynamics integrator
- Lap-level behavior

Implementation files:

- validation/validation_framework.py
- validation/calibration.py
- validation/report_generator.py
- validation/run_validation.py
- validation/validation_plots.py

## Validation Hierarchy

1. Component level:
- Tire force curves, thermal response, degradation behavior.
- Aero downforce/drag curves, mode deltas, ground effect sensitivity.
- Power output, split compliance, regen capability.

2. Integration level:
- Acceleration, braking, cornering, top speed from integrated model.
- Cross-subsystem state consistency (fuel, SOC, loads, slip).

3. System level:
- Coarse lap-time simulation envelope against known track pace.

## Reference Sources

- FIA regulations and technical constraints.
- Publicly available lap and speed metrics.
- Broadcaster telemetry summaries.
- Physics-consistency constraints from literature.

## Pass/Fail Criteria

Typical tolerances:

- Lap time: +/-10% to +/-15% depending on controller simplicity.
- Acceleration: +/-10%.
- Top speed: +/-5% to +/-10% envelope.
- G-forces: +/-15%.
- Power output: +/-5%.
- Downforce and drag: +/-10%.

Each test has explicit pass/fail boundaries and emits actionable details.

## Statistical Metrics

Reported error metrics include:

- MAE
- RMSE
- R2
- Confidence score per subsystem

Confidence combines pass ratio and issue-count penalties.

## Sensitivity Analysis

Finite-difference local sensitivity is computed for selected high-impact parameters (tire and aero). The report includes:

- Local gradients
- Top sensitive parameters by absolute gradient

This helps prioritize calibration and uncertainty reduction.

## Calibration Process

Calibration is data-driven:

- Primary solver: L-BFGS-B (SciPy when available)
- Fallback: bounded random search

Calibrated groups:

- Tire lateral Pacejka parameters
- Aero high-downforce coefficients
- Powertrain max power limits

## Continuous Validation

Validation should run automatically after model changes. Suggested integration points:

- Local pre-merge checks
- CI pipeline stage before training runs
- Nightly full calibration and report publishing

## Output Artifacts

- docs/validation_report.md
- docs/validation_report.html
- validation/validation_results.json
- validation/calibration_results.json
- validation/plots/*.png

These artifacts are designed for both human review and automated gating.
