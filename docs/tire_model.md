# Tire Model (Pacejka Magic Formula)

## Overview

The tire model in [vehicle_dynamics/tire_model.py](vehicle_dynamics/tire_model.py) implements a Pacejka-style semi-empirical force model designed for high-performance open-wheel simulation. It supports:

- Pure lateral force $F_y$
- Pure longitudinal force $F_x$
- Combined slip reduction
- Surface/core temperature dynamics
- Wear and degradation with compound-specific cliff behavior
- Pressure sensitivity and rolling resistance
- Self-aligning torque $M_z$

The implementation is configuration-driven through [configs/tire_config.json](configs/tire_config.json).

## Core Equations

### Pure Slip Pacejka

$$
F = D \sin\left(C\arctan\left(Bx - E\left(Bx - \arctan(Bx)\right)\right)\right)
$$

- For lateral force: $x = \alpha$
- For longitudinal force: $x = \kappa$

### Load Sensitivity

$$
\mu(F_z) = \mu_0\left(1 - k_\mu \frac{F_z - F_{z,nom}}{F_{z,nom}}\right)
$$

$$
D = \mu(F_z) F_z \cdot M_{grip}
$$

Where $M_{grip}$ combines temperature, wear, pressure, and damage multipliers.

### Combined Slip

Pure forces are scaled by weighting functions:

$$
F_x^{comb} = F_x^{pure} G_{x\alpha}(\alpha), \quad
F_y^{comb} = F_y^{pure} G_{y\kappa}(\kappa)
$$

Then constrained with an ellipse-style limiter for plausibility.

### Temperature Dynamics

Surface temperature uses a lumped energy balance:

$$
\dot{T}_{surf} = \frac{Q_{slip} + Q_{track} - Q_{conv} - Q_{cond}}{m_{surf} c_p}
$$

Core temperature responds slower via conduction:

$$
\dot{T}_{core} = \frac{Q_{cond}}{m_{core} c_p}
$$

### Degradation

Wear is monotonic and slip/temperature dependent:

$$
\dot{w} = k_{wear}\left(|\alpha|^{n_{lat}} + |\kappa|^{n_{long}}\right) M_T
$$

Grip loss combines linear degradation and compound-specific cliff.

### Self-Aligning Torque

$$
M_z = -t_p F_y
$$

with pneumatic trail $t_p$ decaying as slip angle increases.

## Compound Behavior

- Soft: highest peak grip, fastest warm-up, fastest wear, strongest cliff.
- Medium: balanced envelope.
- Hard: lowest peak grip, slower warm-up, longest life.

## API Summary

Main methods in [vehicle_dynamics/tire_model.py](vehicle_dynamics/tire_model.py):

- `calculate_forces(...) -> (Fx, Fy, Mz)`
- `calculate_lateral_force(...)`
- `calculate_longitudinal_force(...)`
- `calculate_combined_forces(...)`
- `update_temperature(...)`
- `update_wear(...)`
- `get_grip_multiplier(...)`
- `get_state()` / `set_state(...)`
- `reset(compound=...)`

## Validation Method

Use [tests/test_tire_model.py](tests/test_tire_model.py) and verify:

- Force magnitudes in expected F1 range
- Load sensitivity trend
- Combined slip reduction
- Thermal rise under sustained slip power
- Monotonic wear increase
- Compound ordering (Soft > Medium > Hard grip)
- No NaN/Inf for edge cases

## Assumptions and Limitations

- Coefficients are representative and tunable, not fitted to proprietary team telemetry.
- Relaxation length and transient carcass dynamics are not yet modeled.
- Contact patch dynamics are approximated through multipliers.
- Thermal model is lumped (surface/core) with a synthetic 3-zone visualization profile.

## References

- Hans B. Pacejka, *Tire and Vehicle Dynamics*.
- Milliken & Milliken, *Race Car Vehicle Dynamics*.
- Public F1 technical analyses for 2022+ ground effect era behavior.
