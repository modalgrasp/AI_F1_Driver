# Yas Marina Circuit Documentation for RL Training

## Track Profile
1. Name: Yas Marina Circuit (Abu Dhabi)
2. FIA Length: 5.281 km
3. Turns: 16 (9 right, 7 left)
4. DRS Zones: 2
5. Benchmark Lap: 1:22.109 (Max Verstappen, 2021)

## Sector Breakdown
### Sector 1 (Turns 1-5)
1. High-speed approach into heavy braking at Turn 1
2. Medium-flowing direction changes requiring stable entry speed
3. Key challenge: brake release precision to avoid understeer

### Sector 2 (Turns 6-11)
1. Long straights with overtaking opportunities and DRS relevance
2. Hard braking events and traction-critical exits
3. Key challenge: maximizing top speed while preserving tire model stability

### Sector 3 (Turns 12-16)
1. Technical final sequence and long-radius corners
2. Exit quality strongly affects lap completion pace
3. Key challenge: avoiding line oscillation and curb aggression

## Turn-by-Turn Guidance
1. Turn 1: heavy-braking right-hander, prioritize stable deceleration
2. Turn 2: medium-speed left transition, avoid over-rotation
3. Turn 3: exit setup corner, optimize throttle ramp
4. Turn 4: directional change, balance yaw rate
5. Turn 5: straight setup, minimize steering scrub
6. Turn 6: fast section entry, maintain confidence line
7. Turn 7: major braking zone, overtaking candidate
8. Turn 8: traction-limited exit
9. Turn 9: medium-speed build-up corner
10. Turn 10: linked sequence, precision line required
11. Turn 11: transition corner into straight
12. Turn 12: high-speed commitment corner
13. Turn 13: long loaded corner, tire temperature sensitive
14. Turn 14: setup for final sequence
15. Turn 15: medium-speed directional control
16. Turn 16: final corner to start/finish, crucial exit speed

## Strategy Features
1. DRS/Overspeed opportunities in Sector 2 straights
2. Pit lane speed limits:
   - 80 km/h in practice/qualifying
   - 60 km/h in race
3. Overtaking mainly enabled by strong exits and braking confidence

## AI Challenges
1. Late-braking stability in Turns 1 and 7
2. Exit traction control in low-speed corners
3. Maintaining racing line smoothness in linked corners
4. Avoiding track-limit violations at corner exits

## Data Files in This Project
1. Raw extraction outputs:
   - `data/tracks/yas_marina/extracted/yas_marina_track_data.json`
   - `data/tracks/yas_marina/extracted/yas_marina_waypoints.csv`
   - `data/tracks/yas_marina/extracted/yas_marina_track_arrays.npz`
   - `data/tracks/yas_marina/extracted/yas_marina_track_data.pkl`
2. Analysis outputs:
   - `data/tracks/yas_marina/analysis/yas_marina_analysis.json`
3. Visual outputs:
   - `data/tracks/yas_marina/visualizations/*.png`
   - `data/tracks/yas_marina/visualizations/*.html`
4. RL configuration:
   - `configs/yas_marina_config.json`

## Coordinate System
1. Assetto Corsa uses right-handed coordinates
2. Units:
   - Distance: meters
   - Speed: km/h (converted to m/s for some calculations)
   - Angles: radians unless explicitly labeled in degrees

## Reverse Engineering Notes
`fast_lane.ai` can be binary/proprietary depending on source.
If direct parse is not possible:
1. Use exported telemetry laps to reconstruct line
2. Use map centerline + width envelope as fallback
3. Create an interim spline and refine with driving data
