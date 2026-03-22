# Vehicle Dynamics Integrator (Step 2.4)

## Overview

The integrated model in [vehicle_dynamics/vehicle_model.py](vehicle_dynamics/vehicle_model.py) combines tire, aerodynamics, and powertrain subsystems into a full planar 6-DOF dynamics simulation with roll and pitch states.

Tracked states:

- Global position: $x, y$
- Heading: $\psi$
- Body velocities: $v_x, v_y$
- Yaw rate: $r$
- Accelerations: $a_x, a_y$
- Wheel speeds: 4 corners
- Suspension attitude: roll, pitch and rates
- Tire thermal and wear states (4 tires)
- Powertrain states (RPM, gear, SOC, fuel)

## Coordinate Systems

- Global frame: fixed to ground.
- Body frame: fixed to chassis, $x$ forward and $y$ left.

Velocity transformation:

$$
\dot x = v_x\cos\psi - v_y\sin\psi,\quad
\dot y = v_x\sin\psi + v_y\cos\psi
$$

## Equations of Motion

Longitudinal and lateral body accelerations:

$$
a_x = \frac{F_x}{m} + v_y r
$$

$$
a_y = \frac{F_y}{m} - v_x r
$$

Yaw acceleration:

$$
\dot r = \frac{M_z}{I_z}
$$

Where total forces/moments include:

- Tire forces from all four tires
- Aero drag/downforce and aero yaw moment
- Rear drive force from powertrain wheel torque

## Load Transfer Model

Normal loads are computed from:

- Static distribution from CG location
- Longitudinal transfer: $\Delta F_z = m a_x h / L$
- Lateral transfer: $\Delta F_z = m a_y h / t$
- Aero downforce split front/rear
- Roll contribution from stiffness and anti-roll bars

## Numerical Integration

- RK4 integration for the 6-state planar core.
- Substepping with configurable max step from [configs/vehicle_config.json](configs/vehicle_config.json).
- State clamping and angle wrapping for stability.

## Subsystem Coupling Strategy

- Powertrain is advanced once per substep (stateful).
- RK4 derivatives use that substep drive force as quasi-constant input.
- Tire states (temperature and wear) are updated after the kinematic state update.
- Wheel rotational dynamics are integrated from drive, brake, and tire reaction torques.

## Validation Summary

Validated through [tests/test_vehicle_dynamics.py](tests/test_vehicle_dynamics.py):

- Straight-line acceleration
- Emergency braking
- Steady-state cornering and lateral G generation
- Positive normal loads and load transfer asymmetry
- Finite RK4 states (no NaN/Inf)
- Cross-subsystem state consistency
- Energy plausibility checks
- Performance check under 1 ms at 1 kHz step size

## Integration for RL

Use the `update(dt, steering, throttle, brake, aero_mode)` entry point. The `get_state()` method returns a full dictionary suitable for observation construction and diagnostics logging.
