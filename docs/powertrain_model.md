# Hybrid Powertrain Model (F1 2026)

## Overview

The hybrid powertrain model in [vehicle_dynamics/powertrain.py](vehicle_dynamics/powertrain.py) represents the 2026 regulation intent of near 50:50 ICE/electric peak contribution. It includes:

- Turbocharged 1.6L V6 ICE
- MGU-K motor/generator
- Optional MGU-H support path
- Energy store (battery) with thermal behavior
- Fuel system with flow and capacity limits
- 8-speed transmission with shift latency
- Overtake mode power boost logic

Strategy logic is separated into [vehicle_dynamics/energy_management.py](vehicle_dynamics/energy_management.py).

## Regulatory Targets

- ICE peak power: about 373 kW
- MGU-K peak motor power: about 373 kW
- Combined peak: about 746 kW (about 1000 HP)
- MGU-K harvest capability: up to 350 kW
- Fuel flow cap: 100 kg/h

## Core Models

## ICE and Turbo

ICE torque uses an RPM-shaped polynomial with throttle scaling and turbo boost lag:

$$
T_{ICE}(n) = T_{max}\left(-2.5\hat n^3 + 3.8\hat n^2 + 0.2\hat n\right)\tau_{th}
$$

Turbo pressure follows first-order dynamics:

$$
\dot p_{boost} = \frac{p_{target} - p_{boost}}{\tau}
$$

Power and fuel flow:

$$
P = T\omega, \quad \dot m_f = \min\left(\frac{P}{3600} \cdot BSFC,\, \frac{100}{3600}\right)
$$

## MGU-K

Motoring and harvesting both respect RPM-dependent limits and efficiency maps:

- Motoring: battery discharge supports wheel torque
- Regeneration: braking/lift-off recovers energy into battery

## Battery and Thermal

Battery SOC/energy integrates power flow with charge/discharge limits and a thermal model. Performance factor reduces available power outside the optimal temperature range.

## Power Blending

`balanced` strategy targets 50:50 split when SOC and operating point permit. `max_power` prioritizes deployment. `save_energy` preserves battery.

## API

Main method:

- `calculate_wheel_power(...) -> (wheel_torque_nm, fuel_flow_kg_s, battery_flow_w, state)`

State includes:

- Power split metrics
- Fuel and SOC values
- Overtake state
- Regen torque and wheel torque

## Energy Management

`EnergyManagementSystem` creates deployment/harvest maps and returns strategy decisions based on:

- Track position
- SOC
- Lap mode (`race`, `qualifying`)

## Validation Approach

Use [tests/test_powertrain.py](tests/test_powertrain.py) to validate:

- Combined and split power targets
- Regen charging behavior
- Fuel depletion behavior
- SOC guardrails and thermal bounds
- Numerical stability and performance

## Integration Notes

- Output `wheel_torque_nm` can be passed directly to longitudinal tire/wheel dynamics.
- Fuel mass can feed total vehicle mass in chassis model.
- EMS output strategies can be consumed by race strategy and RL policy conditioning.

## Limitations

- No full cylinder-resolved combustion model.
- No clutch launch model.
- No detailed inverter/motor thermal network yet.
- MGU-H remains optional and simplified for uncertain 2026 final regulations.
