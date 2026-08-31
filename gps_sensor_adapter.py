"""Parser for the recorded GPS fields in ``matrix_imu_gps.zip``.

This module performs no filtering, fusion, safety mathematics, or IMU use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import zipfile


SENSOR_JSONL_PATH = "sensors/matrix_sensors.jsonl"
GPS_SPEED_SOURCE = "GPS"
UNAVAILABLE_SPEED_SOURCE = "UNAVAILABLE"


@dataclass(frozen=True)
class GPSObservation:
    frame_id: int
    matrix_filename: str
    monotonic_ns: int
    speed_kph: Optional[float]
    speed_mps: Optional[float]
    speed_source: str
    valid: bool
    freshness_ms: Optional[float]
    connected: Optional[bool]
    received: Optional[bool]
    fix: Optional[bool]
    satellites: Optional[int]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]
    gps_updated_monotonic_ns: Optional[int]
    error: Optional[str]
    reason: str


def matrix_filename_for_frame(frame_id: int) -> str:
    """Return the archive path for a non-negative integer sensor frame ID."""
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return f"matrix/{frame_id:06d}.npy"


def parse_gps_sensor_record(record: Mapping[str, Any]) -> GPSObservation:
    """Parse one actual ``matrix_sensors.jsonl`` record.

    Explicit health failures make speed unavailable. Freshness is retained as
    a diagnostic; no unvalidated age threshold is imposed.
    """
    if not isinstance(record, Mapping):
        raise TypeError("sensor record must be a mapping")
    frame_id = record.get("frame_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("record frame_id must be a non-negative integer")
    monotonic_ns = record.get("monotonic_ns")
    if (isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int)
            or monotonic_ns < 0):
        raise ValueError("record monotonic_ns must be a non-negative integer")
    matrix_filename = record.get("raw_matrix_file")
    expected_filename = matrix_filename_for_frame(frame_id)
    if matrix_filename != expected_filename:
        raise ValueError(
            f"frame {frame_id} raw_matrix_file must be {expected_filename!r}, "
            f"got {matrix_filename!r}"
        )
    gps = record.get("gps")
    if not isinstance(gps, Mapping):
        gps = {}

    reasons: list[str] = []
    connected = _optional_bool(gps.get("connected"))
    received = _optional_bool(gps.get("received"))
    fix = _optional_bool(gps.get("fix"))
    if connected is not True:
        reasons.append("GPS not explicitly connected")
    if received is not True:
        reasons.append("GPS record not explicitly received")
    if fix is not True:
        reasons.append("GPS fix unavailable")
    error_value = gps.get("error")
    if error_value is not None:
        reasons.append(f"GPS error: {error_value}")

    speed_kph = _optional_finite_float(gps.get("speed_kph"))
    if speed_kph is None:
        reasons.append("GPS speed_kph missing or non-finite")
    elif speed_kph < 0.0:
        reasons.append("GPS speed_kph is negative")

    age_ns = gps.get("age_ns")
    freshness_ms: Optional[float]
    if (isinstance(age_ns, bool) or not isinstance(age_ns, (int, float))
            or not math.isfinite(float(age_ns)) or float(age_ns) < 0.0):
        freshness_ms = None
        reasons.append("GPS age_ns missing, negative, or non-finite")
    else:
        freshness_ms = float(age_ns) / 1e6

    valid = not reasons
    speed_mps = speed_kph / 3.6 if valid and speed_kph is not None else None
    return GPSObservation(
        frame_id=frame_id,
        matrix_filename=matrix_filename,
        monotonic_ns=monotonic_ns,
        speed_kph=speed_kph,
        speed_mps=speed_mps,
        speed_source=GPS_SPEED_SOURCE if valid else UNAVAILABLE_SPEED_SOURCE,
        valid=valid,
        freshness_ms=freshness_ms,
        connected=connected,
        received=received,
        fix=fix,
        satellites=_optional_int(gps.get("satellites")),
        latitude_deg=_optional_finite_float(gps.get("latitude_deg")),
        longitude_deg=_optional_finite_float(gps.get("longitude_deg")),
        gps_updated_monotonic_ns=_optional_int(gps.get("updated_monotonic_ns")),
        error=None if error_value is None else str(error_value),
        reason="valid recorded GPS speed" if valid else "; ".join(reasons),
    )


def load_gps_observations_from_archive(
    archive_path: str | Path,
    sensor_jsonl_path: str = SENSOR_JSONL_PATH,
) -> list[GPSObservation]:
    """Load and validate chronological sensor records without extracting data."""
    with zipfile.ZipFile(archive_path) as archive:
        try:
            lines = archive.read(sensor_jsonl_path).decode("utf-8").splitlines()
        except KeyError as exc:
            raise ValueError(f"archive lacks {sensor_jsonl_path}") from exc
        archive_names = set(archive.namelist())
    observations = [parse_gps_sensor_record(json.loads(line)) for line in lines]
    seen: set[int] = set()
    previous_timestamp_ns: Optional[int] = None
    for observation in observations:
        if observation.frame_id in seen:
            raise ValueError(f"duplicate sensor frame_id {observation.frame_id}")
        seen.add(observation.frame_id)
        if observation.matrix_filename not in archive_names:
            raise ValueError(
                f"sensor frame {observation.frame_id} has no mapped matrix "
                f"{observation.matrix_filename}"
            )
        if (previous_timestamp_ns is not None
                and observation.monotonic_ns <= previous_timestamp_ns):
            raise ValueError("sensor monotonic_ns values must increase strictly")
        previous_timestamp_ns = observation.monotonic_ns
    return observations


def _optional_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _optional_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
