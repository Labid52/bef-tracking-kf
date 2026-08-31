#!/usr/bin/env python3
"""Tests for diagnostic-only IMU longitudinal acceleration calibration."""

import math

import numpy as np

import imu_acceleration_adapter as accel
import imu_sensor_adapter as yaw
from live_bev_viewer import LiveBEVProcessor


def record(acceleration=(0.0, 0.0, 9.80665), roll_deg=0.0,
           pitch_deg=0.0, update_ns=1_000_000_000):
    return {
        "frame_id": 0,
        "raw_matrix_file": "matrix/000000.npy",
        "monotonic_ns": update_ns + 5_000_000,
        "imu": {
            "connected": True,
            "error": None,
            "updated_monotonic_ns": update_ns,
            "age_ns": 5_000_000,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": 20.0,
            "acceleration_mps2": list(acceleration),
        },
    }


def test_actual_acceleration_record_parsing_preserves_components():
    observation = accel.parse_imu_acceleration_record(
        record(acceleration=(-1.0, 0.2, 9.7), roll_deg=1.0, pitch_deg=8.0)
    )
    assert observation.valid
    assert observation.raw_acceleration_components_mps2 == (-1.0, 0.2, 9.7)
    assert observation.freshness_ms == 5.0
    assert observation.yaw_deg == 20.0


def test_missing_malformed_and_nonfinite_acceleration_are_invalid():
    samples = [record(acceleration=(1.0, 2.0)), record(acceleration=(1.0, math.nan, 3.0)), record()]
    del samples[-1]["imu"]["acceleration_mps2"]
    for sample in samples:
        observation = accel.parse_imu_acceleration_record(sample)
        assert not observation.valid
        assert observation.vehicle_longitudinal_acceleration_mps2 is None


def test_timestamp_validation():
    sample = record()
    sample["imu"]["updated_monotonic_ns"] = -1
    observation = accel.parse_imu_acceleration_record(sample)
    assert not observation.valid
    assert "timestamp" in observation.reason or "monotonic" in observation.reason


def test_zero_roll_pitch_gravity_compensates_to_zero():
    corrected = accel.compensate_gravity_mps2(
        (0.0, 0.0, accel.STANDARD_GRAVITY_MPS2), 0.0, 0.0
    )
    assert all(abs(value) < 1e-12 for value in corrected)


def test_nonzero_pitch_synthetic_gravity_compensates_to_zero():
    gravity = accel.gravity_projection_components_mps2(0.0, 10.0)
    corrected = accel.compensate_gravity_mps2(gravity, 0.0, 10.0)
    assert all(abs(value) < 1e-12 for value in corrected)
    assert gravity[0] < 0.0


def test_roll_sign_handling_matches_supported_projection():
    gravity = accel.gravity_projection_components_mps2(10.0, 0.0)
    assert gravity[1] > 0.0
    assert all(abs(value) < 1e-12 for value in
               accel.compensate_gravity_mps2(gravity, 10.0, 0.0))
    wrong_roll = accel.compensate_gravity_mps2(gravity, -10.0, 0.0)
    assert wrong_roll[1] > 3.0


def test_positive_component_zero_is_positive_vehicle_longitudinal():
    gravity = accel.gravity_projection_components_mps2(2.0, 8.0)
    raw = (gravity[0] + 2.0, gravity[1], gravity[2])
    observation = accel.parse_imu_acceleration_record(
        record(raw, roll_deg=2.0, pitch_deg=8.0)
    )
    assert math.isclose(observation.vehicle_longitudinal_acceleration_mps2, 2.0)


def test_stationary_synthetic_observation_is_zero_longitudinal():
    gravity = accel.gravity_projection_components_mps2(-3.0, 12.0)
    observation = accel.parse_imu_acceleration_record(
        record(gravity, roll_deg=-3.0, pitch_deg=12.0)
    )
    assert abs(observation.vehicle_longitudinal_acceleration_mps2) < 1e-12


def test_acceleration_diagnostic_does_not_change_safety_speed_or_output():
    matrix = np.zeros((120, 80), dtype=np.uint8)
    matrix[70, 40] = 1
    baseline = LiveBEVProcessor().update(matrix, dt=0.05, ego_speed_mps=8.0)
    observation = accel.parse_imu_acceleration_record(record())
    assert observation.vehicle_longitudinal_acceleration_mps2 == 0.0
    after_diagnostic = LiveBEVProcessor().update(
        matrix, dt=0.05, ego_speed_mps=8.0
    )
    assert after_diagnostic.safety_result.input_ego_speed_mps == 8.0
    assert after_diagnostic.safety_zone == baseline.safety_zone
    assert after_diagnostic.safe_target_speed_mph == baseline.safe_target_speed_mph
    assert np.array_equal(after_diagnostic.cleaned_matrix, baseline.cleaned_matrix)


def test_yaw_mapping_remains_unchanged():
    assert yaw.IMU_YAW_RATE_COMPONENT_INDEX == 2
    assert yaw.IMU_COMPONENT_TO_TRACKER_SIGN == -1.0
    assert yaw.IMU_YAW_MAPPING_STATUS == "SUPPORTED"


def test_actual_archive_acceleration_is_valid():
    observations = accel.load_imu_acceleration_observations_from_archive(
        "matrix_imu_gps.zip"
    )
    assert len(observations) == 2018
    assert all(item.valid for item in observations)
    assert all(item.vehicle_longitudinal_acceleration_mps2 is not None
               for item in observations)
