"""Diagnostic recorded IMU longitudinal-acceleration adapter.

This module does not feed tracking, zones, or target speed. The archive and
handoff serialize acceleration as ``[ax, ay, az]``. Recorded stationary and GPS
derivative evidence supports component 0 as positive vehicle longitudinal after
the explicit roll/pitch gravity subtraction implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import zipfile

from gps_sensor_adapter import SENSOR_JSONL_PATH, matrix_filename_for_frame


STANDARD_GRAVITY_MPS2 = 9.80665
VEHICLE_LONGITUDINAL_COMPONENT_INDEX = 0
VEHICLE_LONGITUDINAL_COMPONENT_SIGN = 1.0
LONGITUDINAL_ACCELERATION_MAPPING_STATUS = "SUPPORTED"


@dataclass(frozen=True)
class IMUAccelerationObservation:
    frame_id: int
    matrix_filename: str
    frame_monotonic_ns: int
    imu_updated_monotonic_ns: Optional[int]
    freshness_ms: Optional[float]
    acceleration_component_0_mps2: Optional[float]
    acceleration_component_1_mps2: Optional[float]
    acceleration_component_2_mps2: Optional[float]
    roll_deg: Optional[float]
    pitch_deg: Optional[float]
    yaw_deg: Optional[float]
    connected: Optional[bool]
    error: Optional[str]
    valid: bool
    reason: str

    @property
    def raw_acceleration_components_mps2(self) -> tuple[Optional[float], ...]:
        return (
            self.acceleration_component_0_mps2,
            self.acceleration_component_1_mps2,
            self.acceleration_component_2_mps2,
        )

    @property
    def gravity_components_mps2(self) -> Optional[tuple[float, float, float]]:
        if not self.valid:
            return None
        return gravity_projection_components_mps2(self.roll_deg, self.pitch_deg)

    @property
    def gravity_compensated_components_mps2(
        self,
    ) -> Optional[tuple[float, float, float]]:
        if not self.valid:
            return None
        return compensate_gravity_mps2(
            self.raw_acceleration_components_mps2, self.roll_deg, self.pitch_deg
        )

    @property
    def vehicle_longitudinal_acceleration_mps2(self) -> Optional[float]:
        corrected = self.gravity_compensated_components_mps2
        if corrected is None:
            return None
        return (
            VEHICLE_LONGITUDINAL_COMPONENT_SIGN
            * corrected[VEHICLE_LONGITUDINAL_COMPONENT_INDEX]
        )


def parse_imu_acceleration_record(
    record: Mapping[str, Any],
) -> IMUAccelerationObservation:
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
    components: list[Optional[float]] = [None, None, None]
    vector = imu.get("acceleration_mps2")
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        reasons.append("IMU acceleration_mps2 must contain exactly three components")
    else:
        components = [_finite_optional(value) for value in vector]
        if any(value is None for value in components):
            reasons.append("IMU acceleration_mps2 contains a non-finite component")
    roll_deg = _finite_optional(imu.get("roll_deg"))
    pitch_deg = _finite_optional(imu.get("pitch_deg"))
    yaw_deg = _finite_optional(imu.get("yaw_deg"))
    if roll_deg is None:
        reasons.append("IMU roll_deg missing or non-finite")
    if pitch_deg is None:
        reasons.append("IMU pitch_deg missing or non-finite")
    age_ns = imu.get("age_ns")
    if (isinstance(age_ns, bool) or not isinstance(age_ns, (int, float))
            or not math.isfinite(float(age_ns)) or float(age_ns) < 0.0):
        freshness_ms = None
        reasons.append("IMU age_ns missing, negative, or non-finite")
    else:
        freshness_ms = float(age_ns) / 1e6
    valid = not reasons
    return IMUAccelerationObservation(
        frame_id=frame_id,
        matrix_filename=matrix_filename,
        frame_monotonic_ns=frame_ns,
        imu_updated_monotonic_ns=updated_ns,
        freshness_ms=freshness_ms,
        acceleration_component_0_mps2=components[0],
        acceleration_component_1_mps2=components[1],
        acceleration_component_2_mps2=components[2],
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        connected=connected,
        error=None if error_value is None else str(error_value),
        valid=valid,
        reason=(
            "valid diagnostic IMU acceleration observation"
            if valid else "; ".join(reasons)
        ),
    )


def gravity_projection_components_mps2(
    roll_deg: float,
    pitch_deg: float,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> tuple[float, float, float]:
    """Gravity in recorded body components for the empirically supported signs."""
    roll_rad = math.radians(_required_finite("roll_deg", roll_deg))
    pitch_rad = math.radians(_required_finite("pitch_deg", pitch_deg))
    gravity = _required_finite("gravity_mps2", gravity_mps2)
    if gravity <= 0.0:
        raise ValueError("gravity_mps2 must be positive")
    return (
        -gravity * math.sin(pitch_rad),
        gravity * math.sin(roll_rad) * math.cos(pitch_rad),
        gravity * math.cos(roll_rad) * math.cos(pitch_rad),
    )


def compensate_gravity_mps2(
    raw_acceleration_components_mps2: tuple[float, float, float],
    roll_deg: float,
    pitch_deg: float,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> tuple[float, float, float]:
    if len(raw_acceleration_components_mps2) != 3:
        raise ValueError("raw acceleration must contain exactly three components")
    raw = tuple(
        _required_finite(f"acceleration component {index}", value)
        for index, value in enumerate(raw_acceleration_components_mps2)
    )
    gravity = gravity_projection_components_mps2(
        roll_deg, pitch_deg, gravity_mps2
    )
    return tuple(raw[index] - gravity[index] for index in range(3))


def load_imu_acceleration_observations_from_archive(
    archive_path: str | Path,
    sensor_jsonl_path: str = SENSOR_JSONL_PATH,
) -> list[IMUAccelerationObservation]:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            lines = archive.read(sensor_jsonl_path).decode("utf-8").splitlines()
        except KeyError as exc:
            raise ValueError(f"archive lacks {sensor_jsonl_path}") from exc
    observations = [parse_imu_acceleration_record(json.loads(line)) for line in lines]
    if len({item.frame_id for item in observations}) != len(observations):
        raise ValueError("duplicate IMU acceleration frame_id")
    return observations


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
