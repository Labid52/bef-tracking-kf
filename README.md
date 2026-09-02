# Standalone Real-Time Longitudinal Safety

This repository contains a separate, read-only longitudinal-safety consumer.
The upstream perception/sensor program runs normally and requires no changes.
Our process reads its BEV matrices and GPS/IMU records, then internally runs the
causal tracker, nominal target-speed baseline, and V1 longitudinal safety layer.
It displays and logs dynamic `SAFE`, `CRITICAL`, or `COLLISION` zones and the
validated safe target speed.

There is no throttle, brake, steering, CAN, or other actuation output.

## Architecture

```text
Upstream producer (unchanged)
      |
      +-- matrix/*.npy
      +-- matrix_sensors.jsonl
                |
                v
run_real_longitudinal_safety.py
                |
                +-- persistent causal tracker
                +-- nominal target-speed baseline
                +-- dynamic V1 safety zones
                +-- safe target speed
                +-- full 120x80 live BEV window
                +-- streamed CSV log
```

## Clone or update this branch

New clone:

```bash
git clone https://github.com/Labid52/bef-tracking-kf.git
cd bef-tracking-kf
git checkout bev-tracker-20hz-fixed
```

Existing clone:

```bash
git checkout bev-tracker-20hz-fixed
git pull origin bev-tracker-20hz-fixed
```

## Python environment

The runtime requires Python 3, NumPy, SciPy, and OpenCV (`cv2`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The lab development environment currently warns that its SciPy supports NumPy
`<1.25` while NumPy 1.26.4 is installed. All reported tests passed despite that
warning. The compatible ranges in `requirements.txt` avoid this known mismatch
in a clean environment. This work did not alter the lab environment.

## Required upstream outputs

### Matrix directory

The producer must write sequentially numbered NumPy files:

```text
000000.npy
000001.npy
000002.npy
...
```

Each matrix must have:

```text
shape = (120, 80)
dtype = uint8
resolution = 1 m/cell
ego reference = row 80, column 40
```

### Sensor JSONL file

The producer must append one complete JSON object per newline to a file such as
`matrix_sensors.jsonl`, using the schema validated from `matrix_imu_gps.zip`.
A shortened representative record is:

```json
{
  "frame_id": 42,
  "raw_matrix_file": "matrix/000042.npy",
  "monotonic_ns": 123456789000,
  "gps": {
    "connected": true, "received": true, "fix": true,
    "speed_kph": 18.0, "updated_monotonic_ns": 123456789000,
    "age_ns": 0, "error": null
  },
  "imu": {
    "connected": true,
    "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
    "acceleration_mps2": [0.0, 0.0, 9.80665],
    "angular_rate_rps": [0.0, 0.0, 0.0],
    "updated_monotonic_ns": 123456789000,
    "age_ns": 0, "error": null
  }
}
```

The adapters use `speed_mps = speed_kph / 3.6` and tracker yaw rate
`= -angular_rate_rps[2]`. Optional GPS latitude, longitude, and satellite
metadata may also be present.

## Frame matching and sensor assumption

Matrix frame `k` is paired only with sensor record `frame_id == k`. The consumer
supports matrix-first or sensor-first arrival and never substitutes another
frame's record. It waits briefly for the exact pair and protects against partial
matrix writes and incomplete JSONL lines.

At this development stage, a current GPS and IMU record is assumed available
for every BEV frame. Exact frame IDs and monotonic timestamps are still used.
Optional multirate research tools remain separate from this primary live path.

## Run the two processes

Terminal 1:

```text
Run your existing BEV/GPS/IMU producer exactly as usual.
No changes to that producer are required.
```

From its configuration, identify:

```text
<PATH_TO_LIVE_MATRIX_DIRECTORY>
<PATH_TO_MATRIX_SENSORS_JSONL>
```

Terminal 2:

```bash
cd <REPOSITORY_PATH>

python3 run_real_longitudinal_safety.py \
  --matrix-dir <PATH_TO_LIVE_MATRIX_DIRECTORY> \
  --sensor-file <PATH_TO_MATRIX_SENSORS_JSONL>
```

`--matrix-dir` contains numbered `.npy` files. `--sensor-file` is the append-only
sensor JSONL. Startup defaults to the newest available matrix. Use
`--start earliest` or `--start FRAME_ID` only when intentional.

Supported options:

```text
--matrix-dir PATH    matrix directory (default: matrix)
--sensor-file PATH   sensor JSONL (default: sensors/matrix_sensors.jsonl)
--start VALUE        latest, earliest, or integer frame ID
--poll-ms MS         idle polling interval (default: 10 ms)
--headless           run without an OpenCV window
--log PATH           stream CSV to an explicit path
--no-log             disable CSV logging
```

Logging defaults to `logs/realtime_safety_YYYYMMDD_HHMMSS.csv`.

## What the window shows

- Complete incoming 120x80 BEV, including objects outside the corridor and
  behind ego
- Current zone and safe target speed
- Dynamic collision and critical boundaries
- Nearest safety-relevant provisional BEV-origin obstacle range
- GPS speed, IMU yaw rate/source, and diagnostic IMU acceleration
- Observation coverage, raw/tracker dt, and processing timing
- Processed, expired/skipped, malformed, and pending input counters

`COLLISION` means the supplied obstacle range is inside the modeled emergency
stopping boundary. It does **not** mean confirmed physical contact.

## Nominal target versus safe target

```text
target_speed.py -> NOMINAL TARGET SPEED
                       |
                       v
longitudinal safety -> SAFE TARGET SPEED
```

The nominal target can exist when GPS is invalid and V1 cannot evaluate. The
standalone application then reports:

```text
CURRENT ZONE: UNAVAILABLE
NOMINAL TARGET SPEED: <numeric baseline value>
SAFE TARGET SPEED: UNAVAILABLE
```

It never labels nominal pass-through as a validated safe target.

## Input health

For a normal stream, expect:

```text
Skipped/unpaired frames: 0
Malformed sensor lines:  0
```

Nonzero values produce an `INPUT WARNING` and usually indicate wrong paths,
malformed producer output, or mismatched frame IDs. No safety result is
fabricated for an unmatched frame.

## Logging

Rows are streamed directly to disk rather than held indefinitely. Important
columns include frame/timestamp, GPS validity/reason, IMU yaw and acceleration,
provisional obstacle range, boundaries, zone/status, nominal target,
`safe_target_available`, nullable safe target, dt, and processing latency. When
safety is unavailable, `safe_target_speed_mph` is blank while the numeric
nominal target remains in its separate column.

## Stopping

Use Ctrl+C. With the window focused, q or Esc also exits. Final processing and
input-health counters are printed during shutdown.

## Read-only mannequin test

Initial mannequin testing is read-only. The application only reads, tracks,
evaluates, displays, and logs. It does not control throttle, brake, steering, or
CAN.

## Troubleshooting

### No frames appear

Check `--matrix-dir`, verify numbered `.npy` files are being written, and confirm
the default `--start latest` behavior is appropriate.

### No safety result / SAFE TARGET SPEED is UNAVAILABLE

Check GPS connection, receipt, fix, speed, age, and error fields. Unavailable
means no valid V1 evaluation existed; it is not a zero-speed measurement or an
approved nominal target.

### Skipped/unpaired increases

Verify matrix filenames and sensor `frame_id`/`raw_matrix_file` values match and
that both CLI paths refer to the same producer run.

### Malformed sensor lines increases

Verify the producer emits one complete valid JSON object followed by a newline
for every record.

### OpenCV/display failure

Use `--headless` to process and log without a window.

### SciPy/NumPy warning

Prefer a clean virtual environment using `requirements.txt`. Do not broadly
upgrade an established validated environment without rerunning validation.
