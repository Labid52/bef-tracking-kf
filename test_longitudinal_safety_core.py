#!/usr/bin/env python3
"""Synthetic tests for the BEV-independent longitudinal safety core."""

from dataclasses import replace
import math

import longitudinal_safety_core as core


NOMINAL_MPH = 20.0
RANK = {core.SAFE: 0, core.CRITICAL: 1, core.COLLISION: 2}


def evaluate(speed_mps, gap_m, config=None, uncertainty=None, horizon_m=None):
    return core.evaluate_longitudinal_state(
        core.LongitudinalSafetyState(
            speed_mps, gap_m, NOMINAL_MPH, observation_horizon_m=horizon_m
        ),
        config=config,
        uncertainty=uncertainty,
    )


def test_01_zero_speed():
    result = evaluate(0.0, 10.0)
    assert result.zone == core.SAFE
    assert result.maximum_allowable_delay_s is None
    assert result.delay_margin_s is None


def test_02_far_obstacle():
    result = evaluate(10.0, 80.0)
    assert result.zone == core.SAFE
    assert result.safe_target_speed_mph == NOMINAL_MPH


def test_03_critical_boundary_equality():
    _, critical_m = core.dynamic_boundaries_m(8.0)
    assert evaluate(8.0, critical_m).zone == core.CRITICAL


def test_04_collision_boundary_equality():
    collision_m, _ = core.dynamic_boundaries_m(8.0)
    result = evaluate(8.0, collision_m)
    assert result.zone == core.COLLISION and result.safe_target_speed_mph == 0.0


def test_05_monotonic_zone_and_target_as_gap_decreases():
    results = [evaluate(8.0, gap) for gap in range(60, -1, -1)]
    assert [RANK[r.zone] for r in results] == sorted(RANK[r.zone] for r in results)
    speeds = [r.safe_target_speed_mph for r in results]
    assert all(b <= a + 1e-12 for a, b in zip(speeds, speeds[1:]))


def test_06_boundaries_grow_with_speed():
    low = core.dynamic_boundaries_m(2.0)
    high = core.dynamic_boundaries_m(12.0)
    assert high[0] > low[0] and high[1] > low[1]


def test_07_stronger_emergency_braking_reduces_collision_boundary():
    base = core.LongitudinalSafetyConfig()
    weak = replace(base, vehicle_spec=replace(base.vehicle_spec,
                                               emergency_deceleration_mps2=6.0))
    strong = replace(base, vehicle_spec=replace(base.vehicle_spec,
                                                 emergency_deceleration_mps2=9.0))
    assert core.dynamic_boundaries_m(10.0, strong)[0] < core.dynamic_boundaries_m(10.0, weak)[0]


def test_08_weaker_comfortable_braking_expands_critical_boundary():
    base = core.LongitudinalSafetyConfig()
    weak = replace(base, vehicle_spec=replace(base.vehicle_spec,
                                               comfortable_deceleration_mps2=2.0))
    strong = replace(base, vehicle_spec=replace(base.vehicle_spec,
                                                 comfortable_deceleration_mps2=3.0))
    assert core.dynamic_boundaries_m(10.0, weak)[1] > core.dynamic_boundaries_m(10.0, strong)[1]


def test_09_longer_machine_delay_expands_both_boundaries():
    base = core.LongitudinalSafetyConfig()
    longer = replace(base, machine_response_delay_s=0.8)
    before = core.dynamic_boundaries_m(10.0, base)
    after = core.dynamic_boundaries_m(10.0, longer)
    assert after[0] > before[0] and after[1] > before[1]


def test_10_critical_extra_margin_only_expands_critical_boundary():
    base = core.LongitudinalSafetyConfig()
    larger = replace(base, critical_extra_margin_s=0.9)
    before = core.dynamic_boundaries_m(10.0, base)
    after = core.dynamic_boundaries_m(10.0, larger)
    assert after[0] == before[0] and after[1] > before[1]


def _assert_not_less_conservative(base, uncertain):
    assert RANK[uncertain.zone] >= RANK[base.zone]
    assert uncertain.safe_target_speed_mph <= base.safe_target_speed_mph + 1e-12


def test_11_speed_uncertainty_is_conservative():
    _assert_not_less_conservative(
        evaluate(8.0, 30.0),
        evaluate(8.0, 30.0, uncertainty=core.LongitudinalUncertainty(
            speed_uncertainty_mps=2.0)),
    )


def test_12_gap_uncertainty_is_conservative():
    _assert_not_less_conservative(
        evaluate(8.0, 30.0),
        evaluate(8.0, 30.0, uncertainty=core.LongitudinalUncertainty(
            gap_uncertainty_m=5.0)),
    )


def test_13_delay_uncertainty_is_conservative():
    _assert_not_less_conservative(
        evaluate(8.0, 30.0),
        evaluate(8.0, 30.0, uncertainty=core.LongitudinalUncertainty(
            delay_uncertainty_s=0.5)),
    )


def test_14_allowable_delay_decreases_with_gap():
    far = evaluate(10.0, 40.0).maximum_allowable_delay_s
    near = evaluate(10.0, 20.0).maximum_allowable_delay_s
    assert near < far


def test_15_allowable_delay_decreases_with_speed_at_fixed_gap():
    low = evaluate(5.0, 40.0).maximum_allowable_delay_s
    high = evaluate(10.0, 40.0).maximum_allowable_delay_s
    assert high < low


def test_16_negative_allowable_delay_is_preserved():
    result = evaluate(15.0, 5.0)
    assert result.maximum_allowable_delay_s < 0.0
    assert result.delay_margin_s < result.maximum_allowable_delay_s


def test_17_safe_target_never_exceeds_nominal():
    for speed_mps in (0.0, 2.0, 8.0, 15.0):
        for gap_m in range(0, 81):
            result = evaluate(speed_mps, float(gap_m))
            assert 0.0 <= result.safe_target_speed_mph <= NOMINAL_MPH


def test_18_invalid_vehicle_config_state_and_uncertainty_fail():
    cfg = core.LongitudinalSafetyConfig()
    bad_calls = [
        lambda: evaluate(-1.0, 10.0),
        lambda: evaluate(1.0, -1.0),
        lambda: core.evaluate_longitudinal_state(
            core.LongitudinalSafetyState(1.0, 2.0, -1.0)),
        lambda: evaluate(1.0, 2.0, uncertainty=core.LongitudinalUncertainty(
            gap_uncertainty_m=-1.0)),
        lambda: evaluate(1.0, 2.0, config=replace(cfg, machine_response_delay_s=-1.0)),
        lambda: evaluate(1.0, 2.0, config=replace(
            cfg, vehicle_spec=replace(cfg.vehicle_spec,
                                      emergency_deceleration_mps2=-1.0))),
        lambda: evaluate(1.0, 2.0, config=replace(
            cfg, vehicle_spec=replace(cfg.vehicle_spec, wheelbase_m=5.0))),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input did not fail")


def test_19_no_obstacle_is_safe_with_infinite_positive_delay_budget():
    result = evaluate(10.0, None)
    assert result.zone == core.SAFE
    assert math.isinf(result.maximum_allowable_delay_s)
    assert math.isinf(result.delay_margin_s)


def test_20_unbounded_synthetic_behavior_has_full_coverage():
    result = evaluate(10.0, None)
    assert result.zone == core.SAFE
    assert result.coverage_status == core.HORIZON_FULL
    assert result.collision_horizon_covered
    assert result.critical_horizon_covered
    assert math.isinf(result.maximum_allowable_delay_s)


def test_21_no_obstacle_with_finite_horizon_is_not_unbounded():
    result = evaluate(10.0, None, horizon_m=40.0)
    assert result.zone == core.SAFE
    assert result.maximum_allowable_delay_s is None
    assert result.delay_margin_s is None
    assert math.isfinite(result.observable_horizon_maximum_delay_s)
    assert "observed within forward observation horizon" in result.reason


def test_22_critical_boundary_inside_horizon_is_covered():
    result = evaluate(8.0, None, horizon_m=40.0)
    assert result.critical_boundary_m < 40.0
    assert result.critical_horizon_covered
    assert result.coverage_status == core.HORIZON_FULL


def test_23_critical_limited_when_only_collision_boundary_is_covered():
    result = evaluate(8.9408, None, horizon_m=15.0)
    assert result.collision_horizon_covered
    assert not result.critical_horizon_covered
    assert result.coverage_status == core.HORIZON_CRITICAL_LIMITED


def test_24_emergency_limited_when_collision_boundary_is_outside():
    result = evaluate(8.9408, None, horizon_m=5.0)
    assert not result.collision_horizon_covered
    assert not result.critical_horizon_covered
    assert result.coverage_status == core.HORIZON_EMERGENCY_LIMITED


def test_25_35_mph_critical_boundary_fits_in_80m():
    result = evaluate(35.0 * 0.44704, None, horizon_m=80.0)
    assert result.critical_horizon_covered


def test_26_40_mph_critical_boundary_does_not_fit_in_80m():
    result = evaluate(40.0 * 0.44704, None, horizon_m=80.0)
    assert result.collision_horizon_covered
    assert not result.critical_horizon_covered


def test_27_analytic_maximum_critical_speed_satisfies_boundary_equality():
    result = evaluate(0.0, None, horizon_m=80.0)
    speed_mps = result.maximum_critical_horizon_speed_mps
    _, boundary_m = core.dynamic_boundaries_m(speed_mps)
    assert math.isclose(boundary_m, 80.0, rel_tol=0.0, abs_tol=1e-9)


def test_28_speeds_around_analytic_limit_change_coverage():
    maximum_mps = evaluate(0.0, None, horizon_m=80.0).maximum_critical_horizon_speed_mps
    assert evaluate(maximum_mps - 1e-6, None, horizon_m=80.0).critical_horizon_covered
    assert not evaluate(maximum_mps + 1e-6, None, horizon_m=80.0).critical_horizon_covered


def test_29_speed_uncertainty_reduces_maximum_input_speed():
    base = evaluate(0.0, None, horizon_m=80.0)
    uncertain = evaluate(
        0.0,
        None,
        horizon_m=80.0,
        uncertainty=core.LongitudinalUncertainty(speed_uncertainty_mps=1.0),
    )
    assert math.isclose(
        uncertain.maximum_critical_horizon_speed_mps,
        base.maximum_critical_horizon_speed_mps - 1.0,
    )


def test_30_larger_horizon_increases_compatible_speed():
    short = evaluate(0.0, None, horizon_m=60.0)
    long = evaluate(0.0, None, horizon_m=100.0)
    assert long.maximum_collision_horizon_speed_mps > short.maximum_collision_horizon_speed_mps
    assert long.maximum_critical_horizon_speed_mps > short.maximum_critical_horizon_speed_mps


def test_31_invalid_or_inconsistent_horizon_fails():
    bad_calls = [
        lambda: evaluate(1.0, None, horizon_m=-1.0),
        lambda: evaluate(1.0, None, horizon_m=float("nan")),
        lambda: evaluate(1.0, 11.0, horizon_m=10.0),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid horizon did not fail")


def test_32_horizon_below_clearance_has_no_compatible_speed():
    result = evaluate(0.0, None, horizon_m=1.0)
    assert result.coverage_status == core.HORIZON_EMERGENCY_LIMITED
    assert result.maximum_collision_horizon_speed_mps is None
    assert result.maximum_critical_horizon_speed_mps is None
    assert result.collision_horizon_speed_margin_mps is None
