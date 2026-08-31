#!/usr/bin/env python3
"""Tests for recorded GPS parsing and causal replay utilities."""

import json
import math
import zipfile

import gps_sensor_adapter as gps
import replay_gps_longitudinal_safety as replay


def record(frame_id=7, timestamp_ns=2_000_000_000, speed_kph=36.0):
    return {
        "frame_id": frame_id,
        "raw_matrix_file": f"matrix/{frame_id:06d}.npy",
        "monotonic_ns": timestamp_ns,
        "gps": {
            "connected": True,
            "received": True,
            "fix": True,
            "speed_kph": speed_kph,
            "age_ns": 25_000_000,
            "satellites": 12,
            "latitude_deg": 35.0,
            "longitude_deg": -97.0,
            "updated_monotonic_ns": timestamp_ns - 25_000_000,
            "error": None,
        },
    }


def test_kph_conversion_and_valid_record():
    observation = gps.parse_gps_sensor_record(record())
    assert observation.valid
    assert observation.speed_kph == 36.0
    assert observation.speed_mps == 10.0
    assert observation.freshness_ms == 25.0
    assert observation.speed_source == "GPS"


def test_missing_negative_and_nonfinite_speeds_are_unavailable():
    samples = [record(speed_kph=-1.0), record(speed_kph=float("nan")), record()]
    del samples[-1]["gps"]["speed_kph"]
    for sample in samples:
        observation = gps.parse_gps_sensor_record(sample)
        assert not observation.valid
        assert observation.speed_mps is None
        assert observation.speed_source == "UNAVAILABLE"


def test_explicit_health_failure_is_unavailable_not_zero():
    sample = record(speed_kph=0.0)
    sample["gps"]["fix"] = False
    observation = gps.parse_gps_sensor_record(sample)
    assert not observation.valid
    assert observation.speed_mps is None
    assert replay.resolve_speed_input(observation) == (None, "UNAVAILABLE")


def test_frame_mapping_is_exact_and_mismatch_fails():
    assert gps.matrix_filename_for_frame(2017) == "matrix/002017.npy"
    sample = record()
    sample["raw_matrix_file"] = "matrix/000008.npy"
    try:
        gps.parse_gps_sensor_record(sample)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched matrix mapping accepted")


def test_startup_and_recorded_timestamp_timing():
    startup = replay.compute_replay_timing(1_000_000_000, None, 0.05)
    assert startup.raw_sensor_dt_s is None
    assert startup.tracker_dt_s == 0.05
    assert startup.dt_source == "STARTUP_NOMINAL"
    later = replay.compute_replay_timing(1_100_000_000, 1_000_000_000)
    assert math.isclose(later.raw_sensor_dt_s, 0.1)
    assert math.isclose(later.tracker_dt_s, 0.1)
    assert later.dt_source == "RECORDED_TIMESTAMP"
    assert not later.dt_was_clamped


def test_raw_stall_and_tracker_clamp_remain_distinct():
    timing = replay.compute_replay_timing(1_810_000_000, 1_000_000_000)
    assert timing.raw_sensor_dt_s == 0.81
    assert timing.tracker_dt_s == 0.5
    assert timing.dt_was_clamped


def test_noncausal_or_duplicate_timestamp_fails():
    for current in (1_000_000_000, 900_000_000):
        try:
            replay.compute_replay_timing(current, 1_000_000_000)
        except ValueError:
            pass
        else:
            raise AssertionError("non-increasing timestamp accepted")


def test_synthetic_mode_is_explicit_and_independent():
    sample = record()
    sample["gps"]["fix"] = False
    observation = gps.parse_gps_sensor_record(sample)
    speed, source = replay.resolve_speed_input(observation, 8.0)
    assert speed == 8.0
    assert source == "SYNTHETIC"
    assert observation.speed_mps is None


def test_archive_mapping_is_complete_only_for_sensor_records():
    observations = gps.load_gps_observations_from_archive("matrix_imu_gps.zip")
    assert len(observations) == 2018
    assert observations[0].frame_id == 0
    assert observations[-1].frame_id == 2017
    assert observations[-1].matrix_filename == "matrix/002017.npy"
    # No adapter API pairs matrix 002018: only explicit sensor records drive replay.
    assert all(o.frame_id <= 2017 for o in observations)


def test_sensor_record_without_mapped_matrix_fails_clearly(tmp_path):
    archive_path = tmp_path / "missing_matrix.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            gps.SENSOR_JSONL_PATH,
            json.dumps(record(frame_id=9)) + "\n",
        )
    try:
        gps.load_gps_observations_from_archive(archive_path)
    except ValueError as exc:
        assert "has no mapped matrix" in str(exc)
    else:
        raise AssertionError("sensor record without matrix was accepted")


def test_gps_and_tracker_speed_are_separate_live_diagnostics():
    import numpy as np
    from live_bev_viewer import LiveBEVProcessor

    gps_speed_mps = 10.0
    result = LiveBEVProcessor().update(
        np.zeros((120, 80), dtype=np.uint8), dt=0.05,
        ego_speed_mps=gps_speed_mps,
    )
    assert result.safety_input_ego_speed_mps == gps_speed_mps
    assert result.safety_result.ego_speed_mps == gps_speed_mps
    assert result.tracker_ego_speed_mps != gps_speed_mps
