# Validation Report (Example Format)

Generated: 2026-03-22 12:00:00

## Summary

Overall status: 4/5 subsystems passed

### Confidence Scores

- Tire Model: 0.86
- Aerodynamics Model: 0.91
- Powertrain Model: 0.94
- Vehicle Dynamics: 0.78
- Lap Simulation: 0.62

### Example Findings

## ✅ Tire Model: PASS

### ✅ Tire Force Curves
- Peak force MAE: 820 N
- Peak slip in expected range

### ✅ Temperature Sensitivity
- 100C is optimal grip center

## ❌ Vehicle Dynamics: FAIL

### ❌ Top Speed
- Top speed 287 km/h below expected envelope [315, 365] km/h
- Action: adjust aero drag baseline and powertrain deployment strategy

### ✅ Braking Performance
- Max decel: 4.7G
