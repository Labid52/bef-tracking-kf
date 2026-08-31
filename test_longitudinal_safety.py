#!/usr/bin/env python3
"""Unit tests for the headless longitudinal three-zone safety layer."""

from dataclasses import replace
import math

import numpy as np

import longitudinal_safety as ls
from bev_longitudinal_adapter import extract_bev_longitudinal_observation


NOMINAL_MPH = 20.0
EGO_SPEED_MPS = 4.0
TEST_CONFIG = ls.LongitudinalSafetyConfig(
    vehicle_spec=ls.VehicleLongitudinalSpec(
        emergency_deceleration_mps2=4.0,
        comfortable_deceleration_mps2=2.0,
    ),
    standstill_clearance_m=2.0,
    machine_response_delay_s=0.25,
    critical_extra_margin_s=0.25,
)
# At 4 m/s these deliberately give exact raster distances: collision=5 m,
# critical=8 m, making equality semantics unambiguous in matrix-based tests.


def blank(dtype=np.uint8):
    return np.zeros((ls.ROWS, ls.COLS), dtype=dtype)


def obstacle(distance_m, lateral_m=0, cls=1):
    matrix = blank()
    row = ls.EGO_ROW - int(distance_m)
    col = ls.EGO_COL + int(lateral_m)
    matrix[row, col] = np.uint8(cls)
    return matrix


def evaluate(matrix, ego_speed_mps=EGO_SPEED_MPS, config=TEST_CONFIG, bev_config=None):
    return ls.evaluate_longitudinal_safety(
        matrix, ego_speed_mps, NOMINAL_MPH, dt=0.1, config=config,
        bev_config=bev_config,
    )


def test_01_no_obstacle():
    result = evaluate(blank())
    assert result.zone == ls.SAFE
    assert result.obstacle_distance_m is None
    assert result.safe_target_speed_mph == NOMINAL_MPH


def test_02_far_obstacle():
    assert evaluate(obstacle(30)).zone == ls.SAFE


def test_03_at_critical_boundary_is_critical_and_continuous():
    result = evaluate(obstacle(8))
    assert result.critical_boundary_m == 8.0
    assert result.zone == ls.CRITICAL
    assert math.isclose(result.safe_target_speed_mph, NOMINAL_MPH)


def test_04_inside_critical_region():
    result = evaluate(obstacle(7))
    assert result.zone == ls.CRITICAL
    assert 0.0 < result.safe_target_speed_mph <= NOMINAL_MPH


def test_05_at_collision_boundary_is_collision():
    result = evaluate(obstacle(5))
    assert result.collision_boundary_m == 5.0
    assert result.zone == ls.COLLISION
    assert result.safe_target_speed_mph == 0.0


def test_06_inside_collision_region():
    result = evaluate(obstacle(4, cls=1))
    assert result.zone == ls.COLLISION
    assert result.safe_target_speed_mph == 0.0


def test_07_boundaries_increase_with_ego_speed():
    low = ls.dynamic_boundaries_m(2.0)
    high = ls.dynamic_boundaries_m(12.0)
    assert high[0] > low[0] and high[1] > low[1]


def test_08_zero_ego_speed_is_finite_and_sensible():
    collision_m, critical_m = ls.dynamic_boundaries_m(0.0)
    assert collision_m == critical_m == ls.LongitudinalSafetyConfig().standstill_clearance_m
    assert all(math.isfinite(x) and x >= 0.0 for x in (collision_m, critical_m))


def test_09_obstacle_behind_ego_is_ignored():
    assert evaluate(obstacle(-5)).zone == ls.SAFE


def test_10_obstacle_outside_corridor_is_ignored():
    assert evaluate(obstacle(5, lateral_m=3)).zone == ls.SAFE


def test_11_default_has_no_close_car_blind_zone():
    assert evaluate(obstacle(4, cls=3)).zone == ls.COLLISION
    assert evaluate(obstacle(4, cls=1)).zone == ls.COLLISION
    assert evaluate(obstacle(7, cls=3)).zone == ls.CRITICAL


def test_12_explicit_self_artifact_mask_is_geometric_and_car_only():
    mask = ls.SelfVehicleExclusionConfig(
        x_forward_min_m=1.0,
        x_forward_max_m=2.0,
        lateral_half_width_m=1.0,
    )
    bev_config = ls.BEVLongitudinalAdapterConfig(self_vehicle_exclusion=mask)
    assert evaluate(obstacle(2, lateral_m=1, cls=3), bev_config=bev_config).zone == ls.SAFE
    # Immediately outside each configured edge is again an obstacle.
    assert evaluate(obstacle(3, lateral_m=1, cls=3), bev_config=bev_config).zone == ls.COLLISION
    assert evaluate(obstacle(2, lateral_m=2, cls=3), bev_config=bev_config).zone == ls.COLLISION
    # The class-3 perception mask must never hide other physical classes.
    assert evaluate(obstacle(2, lateral_m=1, cls=1), bev_config=bev_config).zone == ls.COLLISION
    assert evaluate(obstacle(2, lateral_m=1, cls=2), bev_config=bev_config).zone == ls.COLLISION


def test_13_monotonic_distance_behavior():
    results = [evaluate(obstacle(distance_m)) for distance_m in range(12, 0, -1)]
    rank = {ls.SAFE: 0, ls.CRITICAL: 1, ls.COLLISION: 2}
    assert [rank[r.zone] for r in results] == sorted(rank[r.zone] for r in results)
    speeds = [r.safe_target_speed_mph for r in results]
    assert all(b <= a + 1e-12 for a, b in zip(speeds, speeds[1:]))


def test_14_increasing_speed_never_makes_zone_less_conservative():
    matrix = obstacle(12)
    rank = {ls.SAFE: 0, ls.CRITICAL: 1, ls.COLLISION: 2}
    zones = [evaluate(matrix, ego_speed_mps=v).zone for v in (0.0, 2.0, 4.0, 8.0)]
    assert [rank[z] for z in zones] == sorted(rank[z] for z in zones)


def test_15_invalid_inputs_fail_clearly():
    bad_calls = [
        lambda: evaluate(blank(), ego_speed_mps=-1.0),
        lambda: evaluate(blank(), ego_speed_mps=float("nan")),
        lambda: evaluate(np.zeros((10, 10), dtype=np.uint8)),
        lambda: evaluate(blank(dtype=np.float32)),
        lambda: ls.evaluate_longitudinal_safety(blank(), 1.0, -1.0),
        lambda: ls.evaluate_longitudinal_safety(blank(), 1.0, 1.0, dt=0.0),
        lambda: ls.dynamic_boundaries_m(1.0, replace(
            TEST_CONFIG,
            vehicle_spec=replace(TEST_CONFIG.vehicle_spec,
                                 emergency_deceleration_mps2=-1.0),
        )),
        lambda: ls.dynamic_boundaries_m(1.0, replace(
            TEST_CONFIG,
            vehicle_spec=replace(TEST_CONFIG.vehicle_spec,
                                 comfortable_deceleration_mps2=5.0),
        )),
        lambda: ls.nearest_longitudinal_obstacle_m(
            blank(),
            ls.BEVLongitudinalAdapterConfig(
                self_vehicle_exclusion=ls.SelfVehicleExclusionConfig(
                    x_forward_min_m=3.0,
                    x_forward_max_m=2.0,
                    lateral_half_width_m=1.0,
                )
            ),
        ),
    ]
    for call in bad_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input did not fail")


def test_16_bev_offset_is_explicit_and_default_preserves_origin_range():
    matrix = obstacle(10, cls=1)
    provisional = ls.evaluate_longitudinal_safety(
        matrix, EGO_SPEED_MPS, NOMINAL_MPH, config=TEST_CONFIG
    )
    calibrated = ls.evaluate_longitudinal_safety(
        matrix,
        EGO_SPEED_MPS,
        NOMINAL_MPH,
        config=TEST_CONFIG,
        bev_config=ls.BEVLongitudinalAdapterConfig(
            bev_origin_to_front_bumper_m=2.0
        ),
    )
    assert provisional.input_obstacle_gap_m == 10.0
    assert calibrated.input_obstacle_gap_m == 8.0
    assert provisional.observation_horizon_m == 80.0
    assert calibrated.observation_horizon_m == 78.0


def test_17_blank_bev_reports_finite_provisional_horizon():
    observation = extract_bev_longitudinal_observation(blank())
    assert observation.bev_origin_obstacle_range_m is None
    assert observation.obstacle_gap_m is None
    assert observation.bev_origin_observation_horizon_m == 80.0
    assert observation.observation_horizon_m == 80.0
    assert not observation.front_bumper_offset_calibrated
    assert "provisional" in observation.horizon_description


def test_18_bev_obstacle_and_horizon_share_explicit_offset():
    provisional = extract_bev_longitudinal_observation(obstacle(10, cls=3))
    calibrated = extract_bev_longitudinal_observation(
        obstacle(10, cls=3),
        ls.BEVLongitudinalAdapterConfig(bev_origin_to_front_bumper_m=2.0),
    )
    assert provisional.obstacle_gap_m == 10.0
    assert provisional.observation_horizon_m == 80.0
    assert calibrated.obstacle_gap_m == 8.0
    assert calibrated.observation_horizon_m == 78.0
    # Default remains conservative: no class-3 self mask has reappeared.
    assert evaluate(obstacle(4, cls=3)).zone == ls.COLLISION


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
