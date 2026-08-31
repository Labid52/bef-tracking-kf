"""Recorded IMU yaw-rate parsing and calibration helpers.

The JSON serializes ``angular_rate_rps`` as an unlabeled three-element vector.
This adapter preserves component indices. Recorded-data calibration establishes
component 2 as the yaw axis; the tracker sign conversion is explicit below.
Acceleration, roll, and pitch are deliberately unused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import zipfile

from gps_sensor_adapter import SENSOR_JSONL_PATH, matrix_filename_for_frame


IMU_YAW_RATE_COMPONENT_INDEX = 2
IMU_COMPONENT_TO_TRACKER_SIGN = -1.0
IMU_YAW_MAPPING_STATUS = "SUPPORTED"


@dataclass(frozen=True)
class IMUYawObservation:
    frame_id: int
    matrix_filename: str
    frame_monotonic_ns: int
    imu_updated_monotonic_ns: Optional[int]
    yaw_deg: Optional[float]
    angular_rate_component_0_rps: Optional[float]
    angular_rate_component_1_rps: Optional[float]
    angular_rate_component_2_rps: Optional[float]
    freshness_ms: Optional[float]
    connected: Optional[bool]
    error: Optional[str]
    valid: bool
    reason: str

    @property
    def angular_rate_components_rps(self) -> tuple[Optional[float], ...]:
        return (
            self.angular_rate_component_0_rps,
            self.angular_rate_component_1_rps,
            self.angular_rate_component_2_rps,
        )

    @property
    def raw_imu_yaw_rate_rps(self) -> Optional[float]:
        if not self.valid:
            return None
        return self.angular_rate_components_rps[IMU_YAW_RATE_COMPONENT_INDEX]

    @property
    def tracker_vehicle_yaw_rate_rps(self) -> Optional[float]:
        raw = self.raw_imu_yaw_rate_rps
        return None if raw is None else IMU_COMPONENT_TO_TRACKER_SIGN * raw


@dataclass(frozen=True)
class YawDerivativeSample:
    start_frame_id: int
    end_frame_id: int
    dt_s: float
    yaw_rate_from_angle_rps: float
    midpoint_angular_rate_components_rps: tuple[float, float, float]


def parse_imu_yaw_record(record: Mapping[str, Any]) -> IMUYawObservation:
    if not isinstance(record, Mapping):
        raise TypeError("sensor record must be a mapping")
    frame_id = record.get("frame_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("record frame_id must be a non-negative integer")
    matrix_filename = record.get("raw_matrix_file")
    expected = matrix_filename_for_frame(frame_id)
    if matrix_filename != expected:
        raise ValueError(f"frame {frame_id} raw_matrix_file must be {expected!r}")
    frame_ns = record.get("monotonic_ns")
    if isinstance(frame_ns, bool) or not isinstance(frame_ns, int) or frame_ns < 0:
        raise ValueError("record monotonic_ns must be a non-negative integer")
    imu = record.get("imu")
    if not isinstance(imu, Mapping):
        imu = {}
    reasons: list[str] = []
    connected = imu.get("connected") if isinstance(imu.get("connected"), bool) else None
    if connected is not True:
        reasons.append("IMU not explicitly connected")
    error_value = imu.get("error")
    if error_value is not None:
        reasons.append(f"IMU error: {error_value}")
    updated_ns = imu.get("updated_monotonic_ns")
    if (isinstance(updated_ns, bool) or not isinstance(updated_ns, int)
            or updated_ns < 0):
        updated_ns = None
        reasons.append("IMU updated_monotonic_ns missing or invalid")
    yaw_deg = _finite_optional(imu.get("yaw_deg"))
    if yaw_deg is None:
        reasons.append("IMU yaw_deg missing or non-finite")
    vector = imu.get("angular_rate_rps")
    components: list[Optional[float]] = [None, None, None]
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        reasons.append("IMU angular_rate_rps must contain exactly three components")
    else:
        components = [_finite_optional(value) for value in vector]
        if any(value is None for value in components):
            reasons.append("IMU angular_rate_rps contains a non-finite component")
    age_ns = imu.get("age_ns")
    if (isinstance(age_ns, bool) or not isinstance(age_ns, (int, float))
            or not math.isfinite(float(age_ns)) or float(age_ns) < 0.0):
        freshness_ms = None
        reasons.append("IMU age_ns missing, negative, or non-finite")
    else:
        freshness_ms = float(age_ns) / 1e6
    valid = not reasons
    return IMUYawObservation(
        frame_id=frame_id,
        matrix_filename=matrix_filename,
        frame_monotonic_ns=frame_ns,
        imu_updated_monotonic_ns=updated_ns,
        yaw_deg=yaw_deg,
        angular_rate_component_0_rps=components[0],
        angular_rate_component_1_rps=components[1],
        angular_rate_component_2_rps=components[2],
        freshness_ms=freshness_ms,
        connected=connected,
        error=None if error_value is None else str(error_value),
        valid=valid,
        reason="valid recorded IMU yaw observation" if valid else "; ".join(reasons),
    )


def load_imu_yaw_observations_from_archive(
    archive_path: str | Path,
    sensor_jsonl_path: str = SENSOR_JSONL_PATH,
) -> list[IMUYawObservation]:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            lines = archive.read(sensor_jsonl_path).decode("utf-8").splitlines()
        except KeyError as exc:
            raise ValueError(f"archive lacks {sensor_jsonl_path}") from exc
    observations = [parse_imu_yaw_record(json.loads(line)) for line in lines]
    if len({item.frame_id for item in observations}) != len(observations):
        raise ValueError("duplicate IMU frame_id")
    return observations


def unwrap_degrees(yaw_degrees: Sequence[float]) -> list[float]:
    """Unwrap degrees using the shortest signed change in [-180, 180)."""
    if not yaw_degrees:
        return []
    first = _required_finite("yaw_degrees[0]", yaw_degrees[0])
    result = [first]
    prior_raw = first
    for index, value in enumerate(yaw_degrees[1:], start=1):
        current_raw = _required_finite(f"yaw_degrees[{index}]", value)
        delta = (current_raw - prior_raw + 180.0) % 360.0 - 180.0
        result.append(result[-1] + delta)
        prior_raw = current_raw
    return result


def distinct_imu_updates(
    observations: Sequence[IMUYawObservation],
) -> list[IMUYawObservation]:
    """Keep the first occurrence of each update timestamp, in causal order."""
    distinct: list[IMUYawObservation] = []
    seen: set[int] = set()
    for observation in observations:
        timestamp = observation.imu_updated_monotonic_ns
        if timestamp is None or timestamp in seen:
            continue
        if distinct and timestamp <= distinct[-1].imu_updated_monotonic_ns:
            raise ValueError("distinct IMU timestamps must increase")
        seen.add(timestamp)
        distinct.append(observation)
    return distinct


def differentiate_yaw_updates(
    observations: Sequence[IMUYawObservation],
) -> list[YawDerivativeSample]:
    """Differentiate unwrapped yaw over distinct IMU updates.

    Each interval is compared with the midpoint (endpoint-average) measured
    angular-rate vector; no future sample is used for replay integration.
    """
    distinct = distinct_imu_updates(observations)
    valid = [item for item in distinct if item.valid]
    unwrapped_deg = unwrap_degrees([item.yaw_deg for item in valid])
    samples: list[YawDerivativeSample] = []
    for index in range(1, len(valid)):
        prior, current = valid[index - 1], valid[index]
        dt_s = (
            current.imu_updated_monotonic_ns - prior.imu_updated_monotonic_ns
        ) / 1e9
        if dt_s <= 0.0:
            raise ValueError("IMU update timestamps must increase")
        yaw_rate_rps = math.radians(unwrapped_deg[index] - unwrapped_deg[index - 1]) / dt_s
        prior_rates = prior.angular_rate_components_rps
        current_rates = current.angular_rate_components_rps
        midpoint = tuple(
            (float(prior_rates[axis]) + float(current_rates[axis])) / 2.0
            for axis in range(3)
        )
        samples.append(YawDerivativeSample(
            prior.frame_id, current.frame_id, dt_s, yaw_rate_rps, midpoint
        ))
    return samples


def select_axis_and_sensor_sign(
    samples: Sequence[YawDerivativeSample],
) -> tuple[int, float, float]:
    """Select the component/sign with maximum positive Pearson correlation."""
    if len(samples) < 2:
        raise ValueError("at least two yaw derivative samples are required")
    expected = [sample.yaw_rate_from_angle_rps for sample in samples]
    candidates = []
    for axis in range(3):
        values = [sample.midpoint_angular_rate_components_rps[axis] for sample in samples]
        correlation = _correlation(values, expected)
        candidates.extend(((correlation, axis, 1.0), (-correlation, axis, -1.0)))
    correlation, axis, sign = max(candidates, key=lambda item: item[0])
    return axis, sign, correlation


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    left_mean, right_mean = sum(left) / n, sum(right) / n
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return numerator / math.sqrt(left_energy * right_energy)


def _finite_optional(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _required_finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
