# Longitudinal Three-Zone Safety V1

## Architecture

The authoritative mathematics in `longitudinal_safety_core.py` is independent
of BEV, tracking, semantic classes, OpenCV, and sensor files:

```text
synthetic or future sensor physical state -> generic safety core
cleaned BEV -> BEV adapter ---------------^
```

`bev_longitudinal_adapter.py` extracts the nearest relevant BEV-origin range.
`longitudinal_safety.py` preserves the original matrix API as a compatibility
wrapper around the adapter and generic core. The frozen tracker and unchanged
`getTargetSpeed()` still provide the cleaned BEV and nominal target.

## Vehicle specification and provisional parameters

Values supplied by the vehicle team:

| Value | Specification |
|---|---:|
| Vehicle width | 1.89 m |
| Vehicle length | 4.66 m |
| Wheelbase | 3.00 m |
| Maximum comfortable braking magnitude | 2.0 m/s² |
| Maximum emergency braking magnitude | 9.0 m/s² |
| Steering/yaw limits | unknown/TBD |

Width, length, and wheelbase are stored for future work but are not used in the
scalar longitudinal equations. They do not determine camera-to-bumper offset.

Provisional test parameters:

| Parameter | Value |
|---|---:|
| Standstill clearance | 2.0 m |
| Machine response delay | 0.30 s |
| Critical extra margin | 0.45 s |
| Critical total intervention time | 0.75 s |
| Bounded uncertainty defaults | all zero |

These are not road-validated parameters. `machine_response_delay_s` represents
the aggregate time before emergency longitudinal response becomes effective;
it may eventually include perception, processing, communication, controller,
and brake-buildup delays. `critical_extra_margin_s` starts proactive intervention
earlier and is not a second independent hardware delay.

`LEGACY_PROVISIONAL_V1_CONFIG` retains the previous 3.0/6.0 m/s² braking values
with equivalent 0.30/0.75 s timing for replay comparison.

## Generic physical API

```python
from longitudinal_safety_core import (
    LongitudinalSafetyState,
    LongitudinalUncertainty,
    evaluate_longitudinal_state,
)

result = evaluate_longitudinal_state(
    LongitudinalSafetyState(
        ego_speed_mps=8.0,
        obstacle_gap_m=25.0,
        nominal_target_speed_mph=20.0,
    ),
    uncertainty=LongitudinalUncertainty(),
)
```

`obstacle_gap_m` is the available longitudinal free gap from the relevant ego
front reference to the obstacle. Synthetic tests supply it directly.
`observation_horizon_m=None` explicitly selects the unbounded synthetic
assumption. A finite horizon means only that forward extent is observed; with
`obstacle_gap_m=None`, the result says no relevant obstacle was **observed
within the horizon**, not that none exists beyond it. A detected obstacle may
not be farther than its declared finite horizon.

Conservative bounded uncertainty uses:

```text
effective speed = input speed + speed uncertainty
effective gap = max(0, input gap - gap uncertainty)
effective machine delay = configured delay + delay uncertainty
```

## Equations and zones

```text
d_collision = standstill_clearance
            + effective_speed * effective_machine_delay
            + effective_speed² / (2 * emergency_deceleration)

d_critical  = standstill_clearance
            + effective_speed * (effective_machine_delay + critical_extra_margin)
            + effective_speed² / (2 * comfortable_deceleration)
```

- `SAFE`: effective gap `>` critical boundary; nominal target passes through.
- `CRITICAL`: collision boundary `<` gap `<=` critical boundary.
- `COLLISION`: gap `<=` collision boundary; target is 0 mph.

CRITICAL preserves the V1 continuous cap:

```text
fraction = (gap - collision_boundary) / (critical_boundary - collision_boundary)
safe_target = nominal_target * sqrt(fraction)
```

The result is clamped to `[0, nominal_target]`.

For positive effective speed:

```text
maximum_allowable_delay =
    (effective_gap - standstill_clearance
     - effective_speed²/(2*emergency_deceleration)) / effective_speed

delay_margin = maximum_allowable_delay - effective_machine_delay
```

Negative values are preserved because they indicate that even zero additional
delay is insufficient under the assumed model. At zero speed the delay
diagnostics are `None`. With no obstacle, obstacle-based allowable delay is
infinite only under the explicit unbounded assumption. A finite horizon instead
provides separate `observable_horizon_maximum_delay_s` and margin diagnostics;
these describe guaranteed observable free space, not an actual obstacle.

Coverage is independent of the three-zone result:

- `FULL`: both critical and collision boundaries fit in the horizon.
- `CRITICAL_LIMITED`: collision fits, but critical does not.
- `EMERGENCY_LIMITED`: even the collision boundary does not fit.

For boundary form `R=d0+vT+v²/(2a)`, the analytic maximum effective speed is
`a*(sqrt(T²+2*(R-d0)/a)-T)`. The reported maximum input speed subtracts the
configured deterministic speed uncertainty. Horizon inadequacy is diagnostic
only in V1; it does not add a zone or cap the target speed.

## BEV compatibility adapter

```python
BEVLongitudinalAdapterConfig(
    corridor_width_m=4.0,
    self_vehicle_exclusion=None,
    bev_origin_to_front_bumper_m=None,
)
```

The roof camera/BEV reference is approximately above the steering-wheel or
dashboard region, but its horizontal distance to the front bumper has not been
measured. Therefore the default adapter output is explicitly a **provisional
BEV-origin obstacle range**, not calibrated front-bumper clearance. It preserves
existing replay numbers. If a measured offset is supplied later, the adapter
uses `max(0, BEV range - offset)` and labels it front-bumper gap.

The 120x80, 1 m/cell raster with ego row 80 supplies a nominal 80 m forward
BEV-origin observation horizon. With no calibrated camera-to-bumper offset,
both the range and horizon remain explicitly provisional. Once an offset is
provided, it is subtracted from both values.

No class-3 self-vehicle exclusion is enabled by default because recorded data
does not establish a validated mask. The legacy broad 6 m mask remains an
explicit comparison option only.

## Live integration and limitations

The viewer continues to show zone, explicit safety ego speed, provisional BEV
range, both boundaries, nominal/safe targets, tracker ego estimate, coverage,
and maximum critical-horizon speed. A blank frame says no obstacle was observed
within the finite horizon. Missing explicit ego speed leaves safety
`UNAVAILABLE` and passes nominal through.

V1 remains longitudinal-only and produces a target speed, not an actuator
command. It has no lateral safety, steering, CBF/QP, ML/RL, probabilistic
uncertainty, or GPS/IMU parsing. GPS speed and calibrated sensor uncertainty can
feed the generic state in the next stage without changing its equations. GPS
and IMU mounting geometry remains future calibration work.

Recorded GPS replay is provided by `gps_sensor_adapter.py` and
`replay_gps_longitudinal_safety.py`. GPS measured speed, an explicit synthetic
test speed, and the tracker-internal ego estimate remain distinct. Recorded
timestamp intervals drive tracker replay but never replace the provisional
machine-response delay. See `GPS_LONGITUDINAL_SAFETY_REPLAY.md` for the exact
archive schema, validity policy, synchronized range, and measured replay
statistics. IMU fields remain unused.

## Synthetic and regression commands

```bash
python3 run_longitudinal_safety_sweep.py
python3 run_longitudinal_safety_sweep.py --observation-horizon-m 80
python3 run_longitudinal_safety_sweep.py --speed-uncertainty-mps 1 \
    --gap-uncertainty-m 2 --delay-uncertainty-s 0.2
python3 -m pytest test_longitudinal_safety_core.py -q
python3 test_longitudinal_safety.py
python3 -m pytest test_longitudinal_safety.py test_live_safety_integration.py -q
python3 test_tracker_regression.py
python3 validate_tracker.py --matrix-dir realtime_capture/matrix --quick
```
