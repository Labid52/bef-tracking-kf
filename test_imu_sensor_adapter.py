#!/usr/bin/env python3
"""Tests for recorded IMU yaw-rate parsing and external tracker plumbing."""

import math

import numpy as np

import imu_sensor_adapter as imu
from live_bev_viewer import LiveBEVProcessor
import replay_gps_longitudinal_safety as replay


def record(frame_id=0, update_ns=1_000_000_000, yaw_deg=10.0,
           rates=(0.1, 0.2, 0.3)):
    return {
        "frame_id": frame_id,
        "raw_matrix_file": f"matrix/{frame_id:06d}.npy",
        "monotonic_ns": update_ns + 5_000_000,
        "imu": {
            "connected": True,
            "error": None,
            "updated_monotonic_ns": update_ns,
            "age_ns": 5_000_000,
            "yaw_deg": yaw_deg,
            "angular_rate_rps": list(rates),
        },
    }


def test_actual_record_parsing_and_rad_per_second_preserved():
    observation = imu.parse_imu_yaw_record(record())
    assert observation.valid
    assert observation.angular_rate_components_rps == (0.1, 0.2, 0.3)
    assert observation.raw_imu_yaw_rate_rps == 0.3
    assert observation.tracker_vehicle_yaw_rate_rps == -0.3
    assert observation.freshness_ms == 5.0


def test_missing_nonfinite_or_wrong_length_rate_is_invalid_not_zero():
    samples = [record(rates=(0.1, 0.2)), record(rates=(0.1, math.nan, 0.3))]
    for sample in samples:
        observation = imu.parse_imu_yaw_record(sample)
        assert not observation.valid
        assert observation.raw_imu_yaw_rate_rps is None
        assert observation.tracker_vehicle_yaw_rate_rps is None


def test_yaw_unwrap_across_both_common_wrap_directions():
    assert imu.unwrap_degrees([179.0, -179.0, -178.0]) == [179.0, 181.0, 182.0]
    assert imu.unwrap_degrees([359.0, 1.0, 2.0]) == [359.0, 361.0, 362.0]
    assert imu.unwrap_degrees([-179.0, 179.0]) == [-179.0, -181.0]


def test_differentiation_uses_distinct_updates_and_skips_repeats():
    observations = [
        imu.parse_imu_yaw_record(record(0, 1_000_000_000, 179.0, (0, 0, 0.2))),
        imu.parse_imu_yaw_record(record(1, 1_000_000_000, 179.0, (0, 0, 0.2))),
        imu.parse_imu_yaw_record(record(2, 2_000_000_000, -179.0, (0, 0, 0.2))),
    ]
    distinct = imu.distinct_imu_updates(observations)
    assert [item.frame_id for item in distinct] == [0, 2]
    samples = imu.differentiate_yaw_updates(observations)
    assert len(samples) == 1
    assert math.isclose(samples[0].yaw_rate_from_angle_rps, math.radians(2.0))


def test_axis_and_sign_calibration_helper_selects_component_two_positive():
    observations = [
        imu.parse_imu_yaw_record(record(0, 0, 0.0, (0.3, 0.1, 0.0))),
        imu.parse_imu_yaw_record(record(1, 1_000_000_000, 10.0, (-0.2, 0.0, 0.2))),
        imu.parse_imu_yaw_record(record(2, 2_000_000_000, 30.0, (0.1, -0.1, 0.5))),
        imu.parse_imu_yaw_record(record(3, 3_000_000_000, 60.0, (-0.1, 0.2, 0.7))),
    ]
    axis, sign, correlation = imu.select_axis_and_sensor_sign(
        imu.differentiate_yaw_updates(observations)
    )
    assert axis == 2
    assert sign == 1.0
    assert correlation > 0.99


def test_actual_archive_has_supported_mapping_and_no_stale_updates():
    observations = imu.load_imu_yaw_observations_from_archive("matrix_imu_gps.zip")
    assert len(observations) == 2018
    assert all(item.valid for item in observations)
    assert len(imu.distinct_imu_updates(observations)) == 2018
    axis, sign, correlation = imu.select_axis_and_sensor_sign(
        imu.differentiate_yaw_updates(observations)
    )
    assert (axis, sign) == (2, 1.0)
    assert correlation > 0.99


def test_external_yaw_is_explicit_and_gps_speed_remains_safety_speed():
    matrix = np.zeros((120, 80), dtype=np.uint8)
    gps_speed_mps = 8.0
    external_tracker_yaw_rps = -0.25
    result = LiveBEVProcessor().update(
        matrix, dt=0.05, ego_speed_mps=gps_speed_mps,
        ego_yaw_rate=external_tracker_yaw_rps,
    )
    assert result.safety_result.input_ego_speed_mps == gps_speed_mps
    assert result.ego_yaw_rate_rps == external_tracker_yaw_rps
    # The cleaner may retain the supplied GPS value for annotation; the fields
    # and source roles remain explicit even when their numbers happen to match.
    assert result.safety_input_ego_speed_mps == gps_speed_mps
    assert result.tracker_ego_speed_mps == result.ego_speed_mps


def test_default_live_mode_still_uses_tracker_estimate():
    result = LiveBEVProcessor().update(
        np.zeros((120, 80), dtype=np.uint8), dt=0.05, ego_speed_mps=8.0
    )
    assert result.ego_yaw_rate_rps == 0.0
    assert result.safety_result.input_ego_speed_mps == 8.0


def test_replay_yaw_source_is_explicit_and_uses_only_current_observation():
    current = imu.parse_imu_yaw_record(record(rates=(1.0, 2.0, 0.25)))
    assert replay.resolve_yaw_rate_input(current, False) == (
        None, "TRACKER_ESTIMATE"
    )
    assert replay.resolve_yaw_rate_input(current, True) == (-0.25, "IMU")
    invalid_record = record(rates=(1.0, 2.0, math.nan))
    invalid = imu.parse_imu_yaw_record(invalid_record)
    assert replay.resolve_yaw_rate_input(invalid, True) == (
        None, "TRACKER_ESTIMATE"
    )
