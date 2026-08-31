#!/usr/bin/env python3
"""Integration tests for safety inputs and the persistent live wrapper."""

import numpy as np

from live_bev_viewer import LiveBEVProcessor


def blank():
    return np.zeros((120, 80), dtype=np.uint8)


def test_explicit_safety_speed_is_preserved_without_external_yaw_rate():
    supplied_speed_mps = 8.9408
    result = LiveBEVProcessor().update(
        blank(),
        dt=0.05,
        ego_speed_mps=supplied_speed_mps,
    )
    assert result.safety_result is not None
    assert result.safety_result.ego_speed_mps == supplied_speed_mps
    assert result.safety_input_ego_speed_mps == supplied_speed_mps
    assert result.tracker_ego_speed_mps == result.ego_speed_mps
    # A blank first frame makes the separation observable: the tracker has no
    # scene-flow speed evidence, while safety still uses the explicit input.
    assert result.tracker_ego_speed_mps != supplied_speed_mps
    assert result.obstacle_gap_description == "provisional BEV range"
    assert result.safety_result.observation_horizon_m == 80.0
    assert result.safety_result.maximum_allowable_delay_s is None
    assert result.safety_result.observable_horizon_maximum_delay_s is not None
    assert "observed within forward observation horizon" in result.safety_reason


def test_live_result_exposes_35_vs_40_mph_horizon_coverage():
    at_35 = LiveBEVProcessor().update(
        blank(), dt=0.05, ego_speed_mps=35.0 * 0.44704
    )
    at_40 = LiveBEVProcessor().update(
        blank(), dt=0.05, ego_speed_mps=40.0 * 0.44704
    )
    assert at_35.safety_result.critical_horizon_covered
    assert at_35.safety_result.coverage_status == "FULL"
    assert not at_40.safety_result.critical_horizon_covered
    assert at_40.safety_result.collision_horizon_covered
    assert at_40.safety_result.coverage_status == "CRITICAL_LIMITED"


def test_missing_safety_speed_is_explicitly_unavailable_and_transparent():
    result = LiveBEVProcessor().update(blank(), dt=0.05)
    assert result.safety_result is None
    assert result.safety_input_ego_speed_mps is None
    assert result.safety_zone == "UNAVAILABLE"
    assert result.safe_target_speed_mph == result.nominal_target_speed_mph
    assert "requires explicit ego_speed_mps" in result.safety_reason
