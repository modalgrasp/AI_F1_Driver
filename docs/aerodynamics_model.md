# Aerodynamics Model (2026 Active Aero)

## Overview

The aerodynamic model in [vehicle_dynamics/aerodynamics.py](vehicle_dynamics/aerodynamics.py) simulates 2026-style active aero with mode-dependent lift and drag coefficients, transition dynamics, ground effect sensitivity, and dirty-air behavior.

Configuration is loaded from [configs/aero_config.json](configs/aero_config.json).

## Features

- High-downforce and low-drag modes
- Smooth actuator transition between modes
- Front/rear downforce split
- Total drag including cooling drag
- Ground effect ride-height multiplier
- Dirty-air downforce reduction with following distance
- Pitch/roll/yaw sensitivity and moments
- Air-density update model via temperature, pressure, humidity

## Core Equations

### Downforce

$$
F_{down} = \frac{1}{2}\rho v^2 A C_L
$$

### Drag

$$
F_{drag} = \frac{1}{2}\rho v^2 A C_D
$$

### Mode Transition

Mode coefficients are interpolated over configured transition time, preventing force discontinuities during switching.

### Ground Effect

A ride-height multiplier boosts downforce near optimal floor clearance and reduces it for excessive height or floor-strike risk at very low ride heights.

### Dirty Air

Downforce multiplier is modeled as a distance-dependent wake loss curve with strongest impact at short following distances.

## API Summary

Main methods in [vehicle_dynamics/aerodynamics.py](vehicle_dynamics/aerodynamics.py):

- `calculate_forces(...) -> (downforce_front, downforce_rear, drag, pitch_moment, roll_moment, yaw_moment)`
- `set_aero_mode(mode)`
- `update(dt)`
- `calculate_downforce(...)`
- `calculate_total_drag(...)`
- `calculate_air_density(...)`
- `dirty_air_multiplier(...)`
- `ground_effect_multiplier(...)`
- `get_state()` / `reset()`

## 2026 Regulation Mapping

- Low-drag mode: reduced $|C_L|$, reduced $C_D$ for straights and energy efficiency.
- High-downforce mode: increased $|C_L|$, increased $C_D$ for corner performance.
- Transition lag: aerodynamic state cannot switch instantaneously.

## Validation Method

Use [tests/test_aerodynamics.py](tests/test_aerodynamics.py) to verify:

- $v^2$ scaling of force magnitudes
- Mode-force hierarchy (high-downforce > low-drag downforce)
- Ground effect trend with ride height
- Dirty-air reduction behavior
- Smooth transition step-to-step
- Numerically finite moments and drag

## Assumptions and Limitations

- Coefficients are representative and tuning-oriented, not CFD/wind-tunnel fitted.
- Porpoising is not explicitly oscillatory yet (can be added as stateful instability model).
- Yaw and crosswind effects are simplified through low-order terms.
- Cooling drag model is reduced-order for simulation speed.

## References

- FIA technical summaries for upcoming active aero direction.
- Katz, *Race Car Aerodynamics*.
- Public F1 performance analyses for downforce/drag ranges.
