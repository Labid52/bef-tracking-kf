"""Standalone real-time file consumer for the validated V1 safety pipeline.

This module is read-only with respect to the upstream producer. It pairs an
exact frame-ID matrix and complete JSONL record, then calls one persistent
LiveBEVProcessor. It contains no actuation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import time
from typing import Mapping, Optional

import numpy as np

import temporal_matrix_cleaner as tmc
from gps_sensor_adapter import GPSObservation, parse_gps_sensor_record
from imu_acceleration_adapter import (
    IMUAccelerationObservation, parse_imu_acceleration_record,
)
from imu_sensor_adapter import IMUYawObservation, parse_imu_yaw_record
from live_bev_viewer import LiveBEVProcessor, LiveFrameResult


GPS = "GPS"
GPS_UNAVAILABLE = "GPS_UNAVAILABLE"
IMU = "IMU"
TRACKER_ESTIMATE = "TRACKER_ESTIMATE"
STARTUP_NOMINAL = "STARTUP_NOMINAL"
RECORDED_TIMESTAMP = "RECORDED_TIMESTAMP"


@dataclass(frozen=True)
class RealTimeFrameResult:
    frame_id: int
    sensor_timestamp_ns: int
    matrix_path: Path
    raw_matrix: np.ndarray
    live_result: LiveFrameResult
    gps: GPSObservation
    imu_yaw: IMUYawObservation
    imu_acceleration: IMUAccelerationObservation
    speed_source: str
    yaw_rate_source: str
    raw_dt_s: Optional[float]
    tracker_dt_s: float
    dt_clamped: bool
    processing_time_ms: float

    @property
    def safety_available(self) -> bool:
        return self.live_result.safety_result is not None

    def log_row(self) -> dict[str, object]:
        safety = self.live_result.safety_result
        return {
            "frame": self.frame_id,
            "timestamp_ns": self.sensor_timestamp_ns,
            "gps_speed_mps": self.gps.speed_mps,
            "gps_valid": self.gps.valid,
            "speed_source": self.speed_source,
            "imu_yaw_rate_rps": self.imu_yaw.tracker_vehicle_yaw_rate_rps,
            "yaw_rate_source": self.yaw_rate_source,
            "imu_longitudinal_acceleration_mps2": (
                self.imu_acceleration.vehicle_longitudinal_acceleration_mps2),
            "imu_acceleration_valid": self.imu_acceleration.valid,
            "bev_obstacle_gap_m": _safety(safety, "input_obstacle_gap_m"),
            "collision_boundary_m": _safety(safety, "collision_boundary_m"),
            "critical_boundary_m": _safety(safety, "critical_boundary_m"),
            "zone": self.live_result.safety_zone,
            "coverage_status": _safety(safety, "coverage_status"),
            "nominal_target_speed_mph": self.live_result.nominal_target_speed_mph,
            "safe_target_speed_mph": self.live_result.safe_target_speed_mph,
            "raw_dt_s": self.raw_dt_s,
            "tracker_dt_s": self.tracker_dt_s,
            "dt_clamped": self.dt_clamped,
            "tracker_latency_ms": self.live_result.tracker_latency_ms,
            "processing_time_ms": self.processing_time_ms,
        }


class RealLongitudinalSafetyRunner:
    """One persistent tracker + target-speed + V1 safety pipeline."""

    def __init__(self, nominal_dt_s: float = 0.05) -> None:
        if not math.isfinite(nominal_dt_s) or nominal_dt_s <= 0.0:
            raise ValueError("nominal_dt_s must be finite and positive")
        self.nominal_dt_s = float(nominal_dt_s)
        self.processor = LiveBEVProcessor(nominal_dt=self.nominal_dt_s)
        self.previous_timestamp_ns: Optional[int] = None
        self.last_frame_id: Optional[int] = None
        self.processed_count = 0

    def process_sample(self, matrix: np.ndarray, sensor_record: Mapping,
                       matrix_path: str | Path = "<memory>") -> RealTimeFrameResult:
        started = time.perf_counter()
        raw = np.asarray(matrix)
        if raw.shape != (120, 80):
            raise ValueError(f"matrix must have shape (120, 80), got {raw.shape}")
        if raw.dtype != np.uint8:
            raise TypeError(f"matrix must have dtype uint8, got {raw.dtype}")
        gps = parse_gps_sensor_record(sensor_record)
        imu_yaw = parse_imu_yaw_record(sensor_record)
        imu_acceleration = parse_imu_acceleration_record(sensor_record)
        frame_id = gps.frame_id
        timestamp_ns = gps.monotonic_ns
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            raise ValueError("duplicate or non-increasing frame_id")
        if (self.previous_timestamp_ns is not None
                and timestamp_ns <= self.previous_timestamp_ns):
            raise ValueError("sensor monotonic_ns must increase strictly")
        raw_dt_s = (None if self.previous_timestamp_ns is None else
                    (timestamp_ns - self.previous_timestamp_ns) / 1e9)
        if raw_dt_s is not None and (not math.isfinite(raw_dt_s) or raw_dt_s <= 0.0):
            raise ValueError("raw dt must be finite and positive")
        supplied_dt_s = self.nominal_dt_s if raw_dt_s is None else raw_dt_s
        tracker_dt_s = min(max(supplied_dt_s, tmc.DT_MIN), tmc.DT_MAX)
        speed_mps = gps.speed_mps if gps.valid else None
        yaw_rate_rps = (
            imu_yaw.tracker_vehicle_yaw_rate_rps if imu_yaw.valid else None)
        result = self.processor.update(
            raw, dt=supplied_dt_s, ego_speed_mps=speed_mps,
            ego_yaw_rate=yaw_rate_rps)
        self.previous_timestamp_ns = timestamp_ns
        self.last_frame_id = frame_id
        self.processed_count += 1
        return RealTimeFrameResult(
            frame_id=frame_id, sensor_timestamp_ns=timestamp_ns,
            matrix_path=Path(matrix_path), raw_matrix=raw,
            live_result=result, gps=gps, imu_yaw=imu_yaw,
            imu_acceleration=imu_acceleration,
            speed_source=GPS if gps.valid else GPS_UNAVAILABLE,
            yaw_rate_source=IMU if imu_yaw.valid else TRACKER_ESTIMATE,
            raw_dt_s=raw_dt_s, tracker_dt_s=result.dt_s,
            dt_clamped=not math.isclose(tracker_dt_s, supplied_dt_s,
                                        rel_tol=0.0, abs_tol=1e-15),
            processing_time_ms=(time.perf_counter() - started) * 1e3,
        )


class CompleteJSONLTailer:
    """Incrementally parse newline-terminated records; retain bounded pending IDs."""

    def __init__(self, path: str | Path, min_frame_id: int = 0,
                 max_pending: int = 4096) -> None:
        self.path = Path(path)
        self.min_frame_id = int(min_frame_id)
        self.max_pending = int(max_pending)
        self.offset = 0
        self.buffer = b""
        self.records: dict[int, dict] = {}
        self.malformed_lines = 0

    def poll(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size < self.offset:  # producer rotated/truncated the stream
            self.offset = 0
            self.buffer = b""
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return
        parts = (self.buffer + chunk).split(b"\n")
        self.buffer = parts.pop()  # incomplete final line is never parsed
        for encoded in parts:
            if not encoded.strip():
                continue
            try:
                record = json.loads(encoded.decode("utf-8"))
                frame_id = record["frame_id"]
                if not isinstance(frame_id, int) or isinstance(frame_id, bool):
                    raise ValueError
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
                self.malformed_lines += 1
                continue
            if frame_id >= self.min_frame_id and frame_id not in self.records:
                self.records[frame_id] = record
        if len(self.records) > self.max_pending:
            for frame_id in sorted(self.records)[:-self.max_pending]:
                del self.records[frame_id]

    def pop(self, frame_id: int) -> Optional[dict]:
        return self.records.pop(frame_id, None)

    def discard_through(self, frame_id: int) -> None:
        for old in [key for key in self.records if key <= frame_id]:
            del self.records[old]


class LiveFileSampleSource:
    """Pair stable matrix files and exact current sensor records by frame ID."""

    def __init__(self, matrix_dir: str | Path, sensor_file: str | Path,
                 start: str | int = "latest", max_pending: int = 4096,
                 pair_wait_s: float = 1.0) -> None:
        self.matrix_dir = Path(matrix_dir)
        self.sensor_file = Path(sensor_file)
        self.matrix_dir.mkdir(parents=True, exist_ok=True)
        existing = self._matrix_paths()
        if isinstance(start, int):
            self.minimum_frame_id = start
        elif start == "earliest":
            self.minimum_frame_id = min(existing, default=0)
        elif start == "latest":
            self.minimum_frame_id = max(existing, default=0)
        else:
            raise ValueError("start must be 'latest', 'earliest', or an integer")
        self.last_processed_frame_id = self.minimum_frame_id - 1
        if not math.isfinite(pair_wait_s) or pair_wait_s < 0.0:
            raise ValueError("pair_wait_s must be finite and non-negative")
        self.pair_wait_s = float(pair_wait_s)
        self.tailer = CompleteJSONLTailer(
            sensor_file, min_frame_id=self.minimum_frame_id,
            max_pending=max_pending)
        self._file_signatures: dict[int, tuple[int, int]] = {}
        self._stable_seen: set[int] = set()
        self._first_seen_s: dict[int, float] = {}
        self.skipped_unpaired_count = 0
        self.last_skipped_frame_id: Optional[int] = None

    def _skip_expired(self, frame_id: int) -> None:
        """Advance past an input that stayed unusable beyond the pairing wait."""
        self.last_processed_frame_id = frame_id
        self.skipped_unpaired_count += 1
        self.last_skipped_frame_id = frame_id
        self._first_seen_s.pop(frame_id, None)
        self.tailer.records.pop(frame_id, None)
        self.tailer.discard_through(frame_id)

    def _matrix_paths(self) -> dict[int, Path]:
        result = {}
        for path in self.matrix_dir.glob("*.npy"):
            try:
                frame_id = int(path.stem)
            except ValueError:
                continue
            result[frame_id] = path
        return result

    def poll_ready(self, limit: Optional[int] = None) -> list[tuple[int, Path, np.ndarray, dict]]:
        self.tailer.poll()
        paths = self._matrix_paths()
        ready = []
        for frame_id in sorted(paths):
            if frame_id <= self.last_processed_frame_id or frame_id < self.minimum_frame_id:
                continue
            path = paths[frame_id]
            first_seen = self._first_seen_s.setdefault(frame_id, time.monotonic())
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            if self._file_signatures.get(frame_id) != signature:
                self._file_signatures[frame_id] = signature
                self._stable_seen.discard(frame_id)
                break
            self._stable_seen.add(frame_id)
            record = self.tailer.records.get(frame_id)
            if record is None:
                if time.monotonic() - first_seen < self.pair_wait_s:
                    break
                self._skip_expired(frame_id)
                continue
            try:
                matrix = np.load(path, allow_pickle=False)
            except (OSError, ValueError, EOFError):
                if time.monotonic() - first_seen < self.pair_wait_s:
                    break
                self._skip_expired(frame_id)
                continue
            if matrix.shape != (120, 80) or matrix.dtype != np.uint8:
                if time.monotonic() - first_seen < self.pair_wait_s:
                    break
                self._skip_expired(frame_id)
                continue
            self.tailer.pop(frame_id)
            self.last_processed_frame_id = frame_id
            self._first_seen_s.pop(frame_id, None)
            self.tailer.discard_through(frame_id)
            ready.append((frame_id, path, matrix, record))
            if limit is not None and len(ready) >= limit:
                break
        return ready


def load_matrix_bytes(payload: bytes) -> np.ndarray:
    matrix = np.load(io.BytesIO(payload), allow_pickle=False)
    if matrix.shape != (120, 80) or matrix.dtype != np.uint8:
        raise ValueError("archive matrix violates 120x80 uint8 contract")
    return matrix


def _safety(safety: object, field: str) -> object:
    return None if safety is None else getattr(safety, field)
