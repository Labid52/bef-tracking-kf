# Standalone Real-Time Longitudinal Safety Readiness

## Scope and readiness result

This repository now contains a read-only, standalone consumer for live BEV
matrix files and the existing `matrix_imu_gps.zip` sensor-record contract. It
runs the persistent causal tracker, existing nominal target-speed calculation,
and accepted V1 longitudinal safety filter in one process. It displays and can
stream-log the safe target; it has no actuation API.

This is software-integration readiness for a controlled read-only test. It is
not physical, road, braking, perception, or safety certification.

## Audit

| Requirement | Initial status | Action | Final status |
|---|---|---|---|
| Standalone consumer | Missing | Added independent CLI and reusable runner | Ready |
| New-frame acquisition | Missing | Added bounded polling of numbered `.npy` files | Ready |
| Matrix/sensor synchronization | Missing | Exact `frame_id` pairing | Ready |
| Persistent tracker | Ready in `LiveBEVProcessor` | Instantiate once in runner | Ready |
| GPS parsing | Ready | Reused validated adapter | Ready |
| IMU parsing | Ready | Reused validated yaw and acceleration adapters | Ready |
| Real dt handling | Partial | Timestamp delta plus visible tracker clamp | Ready |
| Full V1 pipeline | Ready | Reused without equation changes | Ready |
| Full 120x80 visualization | Demo only | Added actual-input renderer | Ready |
| Dynamic-zone overlay | Demo only | Uses result boundaries, corridor only | Ready |
| Safe-target display | Ready in demo | Prominent real-input panel | Ready |
| Streaming logging | Missing | Incremental CSV, no history accumulation | Ready |
| Duplicate prevention | Missing | Strict increasing frame IDs | Ready |
| Partial-file protection | Missing | Stable matrix check, load retry, complete JSON lines | Ready |
| Graceful shutdown | Partial | Ctrl+C, q and Esc cleanup | Ready |
| 20 Hz timing | Unassessed | Full recorded workload benchmark | Ready |
| No upstream modification | Pass | Consumer-side implementation only | Pass |
| No actuation | Pass | No write/control interface added | Pass |

## Producer/consumer architecture

The labmate's existing process remains unchanged. It writes numbered matrices
and newline-terminated sensor records. A separate process watches those outputs:

```text
producer -> matrix/NNNNNN.npy + matrix_sensors.jsonl
         -> exact frame-ID pair in standalone consumer
         -> one persistent LiveBEVProcessor
         -> tracker -> cleaned BEV -> nominal target -> V1 safety
         -> full-BEV display + streamed CSV
```

The consumer never writes to producer paths. Startup defaults to the newest
matrix available, avoiding replay of an old backlog; `--start earliest` or an
integer selects another explicit policy.

## Input contract

Matrices must be NumPy `uint8`, shape `120 x 80`. Geometry remains 1 m/cell,
ego `(row=80, col=40)`, with `x_forward=80-row` and `y_right=col-40`.

Sensor JSON uses the recorded schema: top-level `frame_id`,
`raw_matrix_file`, and `monotonic_ns`; GPS health, `speed_kph`, update time and
age; and IMU orientation, `acceleration_mps2`, `angular_rate_rps`, update time,
age and error. Optional position/satellite metadata are not made mandatory by
the existing adapters. GPS conversion remains `speed_mps=speed_kph/3.6`.
Tracker-coordinate yaw remains `-angular_rate_rps[2]`. Longitudinal
acceleration is parsed and logged diagnostically but does not alter V1.

## Acquisition and failure behavior

Only complete newline-terminated JSON records are parsed. Matrices must have a
stable `(size, modification time)` observation before loading. Incomplete loads
are retried. A matrix is paired only with its identical sensor `frame_id`; a
brief configurable wait handles either write order. A permanently unusable or
unpaired input is counted and skipped after the wait rather than being paired
with a different frame. Processing rejects duplicate/non-increasing IDs and
non-increasing timestamps.

The first frame uses the configured nominal tracker startup dt. Later raw dt is
derived causally from consecutive top-level `monotonic_ns` values. The existing
tracker clamp is retained, while raw dt, tracker dt, and clamp status remain
separate diagnostics.

Invalid GPS never becomes zero speed: V1 safety is marked `UNAVAILABLE` and
nominal target passes through. Invalid IMU yaw invokes the existing tracker
estimate fallback and is labeled separately. No future frame is used.

## Safety and display

The accepted V1 equations and speed filtering remain in their existing modules.
`target_speed.py` supplies the nominal target; the safety result supplies the
final safe target. The renderer displays the actual full incoming matrix,
including objects outside the corridor and behind ego. The collision/critical/
safe shading is restricted to the configured longitudinal corridor and uses
the returned boundary values; it does not duplicate equations. Boundaries past
the 80 m view are explicitly labeled as beyond the horizon.

The panel includes frame/timestamp, GPS speed, IMU yaw source/rate,
diagnostic acceleration, nearest provisional BEV-origin obstacle range,
boundaries, coverage, nominal and safe targets, raw/tracker dt, tracker latency,
and total processing latency.

## Validation results

Recorded chronological emulation processed all 2,018 synchronized archive
frames one at a time. Relative to the accepted replay CSV, GPS input, collision
boundary, critical boundary, zone, and safe target each had zero mismatches
(`1e-12` absolute tolerance).

An incremental producer-thread test wrote 300 frames with alternating
matrix-first and sensor-first ordering, deliberately partial `.npy` writes and
incomplete final JSONL lines. All 300 frames were processed once, in order,
with zero duplicates, skips, expired pairs, malformed complete lines, or
partial-write crashes. The processor identity remained constant.

Recorded full-workload timing (2,018 frames, milliseconds):

| Path | Mean | Median | p95 | p99 | Max | >50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Pipeline internal | 4.094 | 3.076 | 9.678 | 11.278 | 23.524 | 0 |
| Archive acquire + parse + process + log preparation | 4.255 | 3.238 | 9.843 | 11.460 | 23.685 | 0 |
| Renderer only (no window presentation) | 0.753 | 0.741 | 0.927 | 1.026 | 1.844 | 0 |
| Full path including render | 5.015 | 4.011 | 10.757 | 12.369 | 24.696 | 0 |

The longest consecutive non-render over-budget run was zero. GUI presentation
and filesystem behavior on the lab PC still need measurement during the actual
producer test; the table isolates safety-path and image-render construction.

## Logging and memory

CSV rows are written and flushed incrementally. Pending sensor records and
timing samples are bounded. The process does not retain an unlimited frame
history. Without `--no-log`, the default is a timestamped file under `logs/`;
`--log PATH` selects an explicit destination.

## Running the two processes

Terminal 1: run the labmate's existing perception/sensor producer unchanged.

Terminal 2:

```bash
cd /home/labid/Videos/longitudinal_control
python3 run_real_longitudinal_safety.py \
  --matrix-dir <live-matrix-directory> \
  --sensor-file <live-sensor-jsonl>
```

Use `--headless` where no display is available. Stop with Ctrl+C; q and Esc also
close the window. No throttle, brake, steering, CAN, or other vehicle command is
emitted.

## Remaining limitations

- Camera/BEV-origin to front-bumper offset remains uncalibrated; displayed gap
  is a provisional BEV-origin range.
- V1 braking and delay values remain provisional, not vehicle validated.
- Perception accuracy and physical obstacle truth are outside this audit.
- Actual producer atomicity, storage latency, GUI timing, sensor freshness and
  host scheduling must be observed during the controlled lab test.
- There is no actuation. The first mannequin exercise must remain display/log
  only.
