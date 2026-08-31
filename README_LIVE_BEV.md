# Live / Causal BEV Viewer

This adds a **real-time visualization and integration wrapper** around the already validated `OnlineTemporalCleaner`. It does **not** change the accepted tracking method or its parameters.

## What the live path does

The runtime pipeline is:

```text
camera / perception
      -> object detection
      -> distance + BEV point estimation
      -> current 120x80 uint8 semantic matrix
      -> OnlineTemporalCleaner.update(current_matrix)
      -> current stabilized 120x80 matrix + tracked objects
      -> getTargetSpeed()
      -> longitudinal control
                 \
                  -> live BEV display / logging
```

`OnlineTemporalCleaner` is instantiated **once** when the process starts. Each call receives only the **current** matrix. Kalman state, track IDs, covariance, velocity estimates, coast counters, class history, and ego-motion filter state are retained inside the cleaner, so old matrices do not need to be passed again.

There is no RTS backward smoothing, future interpolation, or future tracklet stitching in this live path.

## Rates: 20 Hz live, 10 Hz recorded

| | rate | dt | where |
|---|---|---|---|
| **live deployment (nominal)** | **20 Hz** | **0.05 s** | `control_params.LIVE_DT`, `temporal_matrix_cleaner.NOMINAL_DT` |
| recorded datasets in this repo | 10 Hz | 0.10 s | `control_params.LEGACY_DT`; pass `--dt 0.10` |

The tracker is **rate-aware**: nothing reads a fixed frame interval. Every physical
quantity (prediction, process noise, ego-motion flow baseline, coast duration,
duplicate evidence windows, confirmation window) is computed from the `dt` you
supply, so the same code and the same parameters mean the same *physical* thing
at either rate. Policies that are genuinely about elapsed time are configured in
**seconds**; policies that are about *evidence* (how many detections a track has
seen) remain **observation counts**.

## Minimal live loop

```python
import time
from temporal_matrix_cleaner import OnlineTemporalCleaner
from target_speed import getTargetSpeed
import control_params as cp

cleaner = OnlineTemporalCleaner()      # ONCE, at process start
previous = time.monotonic()

while running:
    matrix = get_new_bev_matrix()      # one new 120x80 uint8 matrix

    now = time.monotonic()
    dt = now - previous                # ACTUAL interval, not a constant
    previous = now

    cleaned, objects = cleaner.update(matrix, dt=dt)

    speed = getTargetSpeed(matrix=cleaned, **cp.TARGET_SPEED_KW)
```

If the vehicle publishes a yaw rate, pass it and the internal estimator is skipped:

```python
cleaned, objects = cleaner.update(matrix, dt=dt, ego_yaw_rate=imu_yaw_rate)
```

**Abnormal `dt` policy.** A measured `dt` outside
`[temporal_matrix_cleaner.DT_MIN, DT_MAX]` = `[0.005, 0.5]` s is clamped into that
range. This is deliberate: a multi-second scheduling stall must not become
multiple seconds of blind extrapolation. `LiveBEVProcessor` counts these in
`dt_clamped`. After a stall the affected tracks simply age out through the normal
coast policy.

**One update per frame.** Call `update()` exactly once per newly received matrix.
Never re-feed a matrix, and never feed a future one.

**Visualization must not gate tracking.** `LiveBEVProcessor.update()` (control
path) is separate from `BEVRenderer.render()` (display). Run the renderer on the
latest snapshot, or skip it entirely under load; the tracker state is unaffected.

## Files

Add this file to the same project directory as `temporal_matrix_cleaner.py`:

```text
live_bev_viewer.py
```

It imports the existing files without modifying them:

```text
temporal_matrix_cleaner.py
control_params.py
target_speed.py
```

## Install display dependency

The tracker already depends on NumPy/SciPy. The live viewer additionally uses OpenCV:

```bash
python3 -m pip install opencv-python
```

On a headless Thor image without a desktop/display server, use the tracker normally and publish/render the visualization on another process or workstation instead of calling `cv2.imshow()`.

## Test now with the recorded matrices

From the project root:

```bash
cd /home/labid/Videos/longitudinal_control
python3 live_bev_viewer.py
```

This replays `matrix/*.npy` at the real dataset cadence of 10 Hz. The files are consumed one at a time. Future matrices are not passed to the cleaner.

Useful commands:

```bash
# Short test window
python3 live_bev_viewer.py --start 1110 --end 1190

# Run as fast as possible instead of sleeping to 10 Hz
python3 live_bev_viewer.py --no-realtime

# Benchmark causal tracking with no GUI
python3 live_bev_viewer.py --headless --no-realtime
```

Window controls:

```text
q or ESC : quit
p        : pause/resume replay
r        : reset all temporal tracker state
```

The right panel uses continuous causal track positions and marks temporary causal coasting with `C`. `S` means a safety pass-through detection was restored. The class color remains constant, so the display does not intentionally blink when an object changes between observed and coasted state.

## Actual live integration on NVIDIA DRIVE Thor

The reusable part is `LiveBEVProcessor`. Create it once:

```python
from live_bev_viewer import LiveBEVProcessor, BEVRenderer

processor = LiveBEVProcessor()
renderer = BEVRenderer()
```

Then, whenever the upstream perception system publishes a new matrix:

```python
def on_new_bev_matrix(raw_matrix, imu_yaw_rate=None, ego_speed_mps=None):
    result = processor.update(
        raw_matrix,
        dt=0.10,
        ego_yaw_rate=imu_yaw_rate,
        ego_speed_mps=ego_speed_mps,
    )

    # Control output: use this immediately.
    cleaned_matrix = result.cleaned_matrix
    tracked_objects = result.objects
    target_speed_mph = result.cleaned_target_speed_mph

    # Optional diagnostics display.
    image = renderer.render(raw_matrix, result)

    return cleaned_matrix, tracked_objects, target_speed_mph, image
```

The upstream system does **not** need to provide previous matrices:

```text
matrix(k-2) ----\
matrix(k-1) ----- internal persistent tracker state
matrix(k)   ----/            + current matrix(k)
                              -> current cleaned output(k)
```

Do **not** create a new processor for every frame:

```python
# WRONG: destroys temporal history every call
def on_new_bev_matrix(raw_matrix):
    processor = LiveBEVProcessor()
    return processor.update(raw_matrix)
```

Create it once and keep it alive for the drive/session.

## Using actual IMU yaw rate

The existing cleaner API supports an external yaw-rate measurement:

```python
result = processor.update(
    raw_matrix,
    dt=0.10,
    ego_yaw_rate=current_imu_yaw_rate_rad_s,
)
```

When the live vehicle already has a reliable yaw rate from IMU/localization, this is preferable to making the tracker infer all ego turning from scene flow. If no yaw rate is supplied, the existing causal fallback estimator remains active.

## Variable real-time cadence

The validated dataset is exactly 10 Hz, so `control_params.DT = 0.10 s`. On the deployed system, if matrix arrival time varies, measure the interval with a monotonic clock and pass the actual `dt`:

```python
now = time.monotonic()
dt = now - previous_time
previous_time = now
result = processor.update(raw_matrix, dt=dt, ego_yaw_rate=imu_yaw_rate)
```

Do not change the tracker methodology merely because the live scheduler has small timing jitter.

## Keep visualization out of the safety-critical control path

The cleaner/control output should not depend on the GUI being healthy. Recommended deployment structure:

```text
perception -> LiveBEVProcessor -> cleaned matrix -> target speed/controller
                         |
                         +-> diagnostic BEV viewer/logger
```

For the first integration test, one process can run both. For the vehicle, if GUI rendering becomes expensive or the Thor runs headless, publish the current raw/cleaned BEV diagnostics to a separate visualization process. Dropping a visualization frame must never stop the cleaner/controller from producing its next output.

## Startup and reset

At the beginning of a new route/session:

```python
processor.reset()
```

Use reset after a perception restart, timestamp discontinuity, or route change where previous object tracks are no longer valid.

## What is safe to push to GitHub

The source repository should contain the code, configuration, tests, and documentation—not the generated dataset/results unless intentionally versioned.

Recommended exclusions:

```gitignore
matrix/
cleaned/
cleaned_causal/
__pycache__/
*.mp4
*.npz
```

Before connecting the output to real longitudinal actuation, run the live module in **shadow mode** on the Thor: log raw matrix, cleaned matrix, tracker latency, raw target speed, and cleaned target speed while the existing controller remains authoritative. Then validate timing and safety behavior on real routes before enabling control authority.

## Current interface contract

Input per update:

```text
shape: 120 x 80
dtype: uint8
rate used in the validated dataset: 10 Hz
optional: ego yaw rate [rad/s], ego speed [m/s]
```

Output per update:

```text
cleaned_matrix          120 x 80 uint8
objects                 current continuous tracked states
raw_target_speed_mph    diagnostic
cleaned_target_speed_mph controller candidate
tracker_latency_ms      diagnostic
```

This is the same causal `OnlineTemporalCleaner.update()` architecture already validated in the project; `live_bev_viewer.py` is only the live I/O/display wrapper around it.
