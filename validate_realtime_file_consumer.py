#!/usr/bin/env python3
"""Exercise the live file consumer against an incremental producer thread."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from real_longitudinal_safety_runner import (
    LiveFileSampleSource,
    RealLongitudinalSafetyRunner,
)


FRAME_COUNT = 300


def sensor_record(frame_id: int) -> dict:
    timestamp_ns = 1_000_000_000 + frame_id * 50_000_000
    return {
        "frame_id": frame_id,
        "raw_matrix_file": f"matrix/{frame_id:06d}.npy",
        "monotonic_ns": timestamp_ns,
        "gps": {
            "connected": True, "received": True, "fix": True,
            "latitude_deg": 0.0, "longitude_deg": 0.0,
            "speed_kph": 18.0, "updated_monotonic_ns": timestamp_ns,
            "age_ns": 0, "satellites": 8, "error": None,
        },
        "imu": {
            "connected": True, "roll_deg": 0.0, "pitch_deg": 0.0,
            "yaw_deg": 0.0, "acceleration_mps2": [0.0, 0.0, 9.81],
            "angular_rate_rps": [0.0, 0.0, 0.01],
            "updated_monotonic_ns": timestamp_ns, "age_ns": 0,
            "error": None,
        },
    }


def npy_bytes(matrix: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, matrix, allow_pickle=False)
    return stream.getvalue()


def producer(matrix_dir: Path, sensor_file: Path) -> None:
    for frame_id in range(FRAME_COUNT):
        matrix = np.zeros((120, 80), dtype=np.uint8)
        matrix[70, 40] = 3
        payload = npy_bytes(matrix)
        line = (json.dumps(sensor_record(frame_id)) + "\n").encode()
        path = matrix_dir / f"{frame_id:06d}.npy"
        if frame_id % 2 == 0:  # matrix first, including an observable partial write
            with path.open("wb") as handle:
                handle.write(payload[: len(payload) // 2])
                handle.flush()
                time.sleep(0.0005)
                handle.write(payload[len(payload) // 2 :])
            time.sleep(0.0005)
            with sensor_file.open("ab") as handle:
                handle.write(line)
        else:  # sensor first, including an incomplete final JSONL line
            with sensor_file.open("ab") as handle:
                handle.write(line[:-1])
                handle.flush()
                time.sleep(0.0005)
                handle.write(b"\n")
            time.sleep(0.0005)
            path.write_bytes(payload)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="realtime-consumer-") as temp:
        root = Path(temp)
        matrix_dir = root / "matrix"
        matrix_dir.mkdir()
        sensor_file = root / "matrix_sensors.jsonl"
        sensor_file.touch()
        source = LiveFileSampleSource(
            matrix_dir, sensor_file, start="earliest", pair_wait_s=2.0)
        runner = RealLongitudinalSafetyRunner()
        processor_identity = id(runner.processor)
        thread = threading.Thread(
            target=producer, args=(matrix_dir, sensor_file), daemon=True)
        thread.start()
        processed: list[int] = []
        deadline = time.monotonic() + 30.0
        while (thread.is_alive() or len(processed) < FRAME_COUNT):
            for frame_id, path, matrix, record in source.poll_ready():
                assert frame_id == record["frame_id"]
                result = runner.process_sample(matrix, record, path)
                assert result.frame_id == frame_id
                processed.append(frame_id)
            if time.monotonic() > deadline:
                raise TimeoutError("incremental producer/consumer validation timed out")
            time.sleep(0.0005)
        thread.join()
        expected = list(range(FRAME_COUNT))
        report = {
            "frames_written": FRAME_COUNT,
            "frames_processed": len(processed),
            "duplicates": len(processed) - len(set(processed)),
            "skipped": sorted(set(expected) - set(processed)),
            "chronological": processed == expected,
            "exact_frame_pairing": processed == expected,
            "processor_persistent": id(runner.processor) == processor_identity,
            "partial_write_crashes": 0,
            "malformed_complete_json_lines": source.tailer.malformed_lines,
            "expired_unpaired_frames": source.skipped_unpaired_count,
        }
        print(json.dumps(report, indent=2))
        if not all((
            report["duplicates"] == 0,
            not report["skipped"],
            report["chronological"],
            report["exact_frame_pairing"],
            report["processor_persistent"],
            report["expired_unpaired_frames"] == 0,
        )):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
