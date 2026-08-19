# BEV Temporal Tracking with Kalman Filtering

This repository implements temporal stabilization and object tracking for a sparse Bird's-Eye-View (BEV) semantic matrix used in longitudinal vehicle control.

The project supports three workflows:

1. **Offline cleaning** for dataset analysis and best-quality trajectory reconstruction.
2. **Causal/online cleaning** that uses only the current frame and previously stored tracker state.
3. **Live BEV visualization** for real-time integration on a platform such as NVIDIA Thor.

The core online tracker combines Kalman filtering, global data association, short-gap coasting, class stabilization, duplicate handling, ego-yaw compensation, and a safety pass-through. The online mode does **not** use future frames.

---

## 1. What this repository does

The upstream perception system produces a semantic BEV matrix. Raw detections may temporarily disappear, jump between cells, flicker between classes, or create fragmented object histories.

This repository inserts a temporal tracking layer between perception and downstream longitudinal control:

```text
Camera / Perception
        |
        v
Object Detection
        |
        v
Distance / BEV Position Estimation
        |
        v
Raw 120 x 80 Semantic Matrix
        |
        v
Temporal Tracking
  - Kalman prediction
  - Hungarian data association
  - Kalman correction
  - short dropout coasting
  - class stabilization
  - duplicate handling
  - safety pass-through
        |
        v
Stabilized Current BEV
        |
        +------> Live BEV Viewer
        |
        +------> Target-Speed Calculation
        |
        +------> Longitudinal Controller
```

The online interface accepts **one current matrix at a time**. Previous frames do not need to be passed again because the tracker stores the necessary temporal state internally.

---

## 2. Repository URL

```bash
git clone https://github.com/Labid52/bef-tracking-kf.git
cd bef-tracking-kf
```

> Note: the repository name currently contains `bef`. If this is intended to mean BEV, the project can still be used exactly as documented here.

---

## 3. Python requirements

The project uses Python 3 and requires the following main packages:

```text
numpy
scipy
matplotlib
opencv-python
```

FFmpeg is recommended if you want to create MP4 comparison videos.

### Recommended virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy matplotlib opencv-python
```

If a `requirements.txt` file is present, use:

```bash
python3 -m pip install -r requirements.txt
```

### Optional FFmpeg installation

Ubuntu / NVIDIA development system:

```bash
sudo apt update
sudo apt install ffmpeg
```

Check installation:

```bash
python3 --version
python3 -c "import numpy, scipy, matplotlib, cv2; print('Python dependencies OK')"
ffmpeg -version
```

---

## 4. Important project timing

The recorded dataset and tracker are configured for:

```text
Frame rate: 10 Hz
Nominal dt: 0.10 s
```

The shared timing source is `control_params.py`.

Do not introduce a separate `dt = 0.05` for this dataset.

---

## 5. BEV input format

Each input frame must be a NumPy array with:

```python
matrix.shape == (120, 80)
matrix.dtype == np.uint8
```

The matrix is sparse: most cells are zero and nonzero cells represent semantic detections.

### Geometry

```text
Rows:             120
Columns:           80
Resolution:         1 m / cell
Ego reference:     row 80, column 40
```

Coordinate conversion:

```python
x_forward_m = 80 - row
y_right_m   = col - 40
```

Therefore:

```text
row < 80  -> in front of the ego vehicle
row = 80  -> ego longitudinal reference
row > 80  -> behind the ego longitudinal reference

col < 40  -> left of ego
col = 40  -> ego lateral center
col > 40  -> right of ego
```

### Detection interpretation

Each nonzero semantic cell is treated as a **point detection**.

In this project, that point is interpreted as the estimated position/distance of the detected object reference point, such as the detected front face. It is **not** treated as the full physical footprint of the vehicle or object.

The tracker therefore stabilizes point detections; it does not invent vehicle width, length, rear boundaries, or occupancy polygons.

---

## 6. Semantic class IDs

The current class mapping is:

| ID | Class |
|---:|---|
| 0 | Empty |
| 1 | Person |
| 2 | Bicycle |
| 3 | Car |
| 4 | Motorcycle |
| 5 | Bus |
| 6 | Truck |
| 7 | Stop sign |
| 8 | Traffic light |
| 9 | Red light |
| 10 | Yellow light |
| 11 | Green light |
| 255 | Unknown |

---

## 7. Main files

```text
temporal_matrix_cleaner.py
    Main tracking implementation.
    Contains the offline tracker and OnlineTemporalCleaner.

control_params.py
    Shared dataset timing and target-speed parameters.

run_temporal_cleaning.py
    Processes a recorded matrix sequence in offline or causal mode.

run_cleaned_target_speed.py
    Streaming/causal target-speed demonstration using one matrix at a time.

target_speed.py
    Existing target-speed policy used downstream of the cleaned matrix.

live_bev_viewer.py
    Real-time BEV display and integration wrapper around OnlineTemporalCleaner.

animate_target_speed_bev_cleaned.py
    Creates recorded raw-vs-cleaned MP4 videos.

compare_raw_cleaned.py
    Quantitative raw/offline/causal evaluation and residual-event audit.

inspect_matrix_geometry.py
    Utility used to inspect and verify BEV geometry.
```

Generated output folders normally include:

```text
cleaned/
cleaned_causal/
```

These should generally not be committed to normal Git history.

---

## 8. Quick start with recorded matrices

If you already have the recorded `.npy` frames, place them in:

```text
matrix/
```

Expected example:

```text
bef-tracking-kf/
├── matrix/
│   ├── 000000.npy
│   ├── 000001.npy
│   ├── 000002.npy
│   └── ...
├── temporal_matrix_cleaner.py
├── run_temporal_cleaning.py
└── ...
```

The original dataset used in development contained 3477 frames, but the scripts can process another sequence as long as the matrix format and timing assumptions are appropriate.

### Sanity-check one matrix

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path

p = sorted(Path('matrix').glob('*.npy'))[0]
m = np.load(p)
print('file :', p)
print('shape:', m.shape)
print('dtype:', m.dtype)
print('nonzero detections:', np.count_nonzero(m))
PY
```

Expected shape:

```text
(120, 80)
```

Expected dtype:

```text
uint8
```

---

## 9. Offline mode

Offline mode is intended for research analysis and best-quality trajectory reconstruction.

It may use information from future frames through operations such as:

```text
forward Kalman filtering
RTS backward smoothing
interior-gap interpolation
tracklet stitching
whole-track analysis
```

Because future observations are allowed, **offline output must not be used directly in a live controller**.

### Run offline cleaning

```bash
python3 run_temporal_cleaning.py
```

Equivalent explicit command:

```bash
python3 run_temporal_cleaning.py \
    --mode offline \
    --out cleaned
```

To use a different input directory:

```bash
python3 run_temporal_cleaning.py \
    --matrix-dir /path/to/matrix \
    --mode offline \
    --out cleaned
```

To test only a limited number of frames:

```bash
python3 run_temporal_cleaning.py \
    --mode offline \
    --limit 300 \
    --out cleaned_test
```

---

## 10. Causal / online mode

Causal mode is the deployment-compatible mode.

It uses only:

```text
current matrix
+
internal tracker state from previous frames
```

It does **not** use:

```text
future matrices
RTS backward smoothing
future tracklet stitching
retroactive corrections based on future data
```

### Run causal cleaning on recorded data

```bash
python3 run_temporal_cleaning.py \
    --mode causal \
    --out cleaned_causal
```

This command is useful for evaluating the online algorithm against a recorded route.

It is still causal even though the source is a folder of recorded files: each frame is processed sequentially using the same online update logic.

---

## 11. Core real-time API

For actual live operation, use `OnlineTemporalCleaner` directly.

Minimal example:

```python
from temporal_matrix_cleaner import OnlineTemporalCleaner
import control_params as cp

tracker = OnlineTemporalCleaner()

# Called once for every new BEV frame
cleaned_matrix, objects = tracker.update(
    current_matrix,
    dt=cp.DT,
)
```

The tracker object must be created **once** and kept alive.

Correct:

```python
tracker = OnlineTemporalCleaner()

while system_is_running:
    current_matrix = get_current_bev_matrix()
    cleaned_matrix, objects = tracker.update(
        current_matrix,
        dt=0.10,
    )
```

Incorrect:

```python
while system_is_running:
    tracker = OnlineTemporalCleaner()   # WRONG: resets history every frame
    cleaned_matrix, objects = tracker.update(current_matrix, dt=0.10)
```

The internal tracker state carries the important information from previous frames, including object states, velocity estimates, covariance, track IDs, missed-frame counters, class history, and ego-motion filtering state.

Therefore, you do **not** pass all previous matrices to every call.

---

## 12. Optional external ego yaw rate

If the deployment platform provides a reliable current yaw rate from IMU/localization, it can be passed directly to the online tracker:

```python
cleaned_matrix, objects = tracker.update(
    current_matrix,
    dt=0.10,
    ego_yaw_rate=current_imu_yaw_rate,
)
```

If an external yaw rate is not supplied, the existing causal fallback estimator is used.

For final vehicle integration, use the actual sensor signal only after confirming units, sign convention, timestamp alignment, and synchronization with the BEV frame.

---

## 13. How the online tracker works

For every new frame:

```text
Existing object tracks
        |
        v
Kalman prediction
"Where should each object be now?"
        |
        v
Current-frame detections
        |
        v
Data association
"Which detection belongs to which existing track?"
        |
        v
Kalman correction
"Update position and velocity using the matched measurement"
        |
        v
Handle unmatched tracks/detections
        |
        +--> short-term coasting for missed detections
        +--> new track from unmatched current detection
        |
        v
Safety pass-through
        |
        v
Current stabilized BEV output
```

### Track state

The tracker maintains a continuous state approximately of the form:

```text
[x, y, vx, vy]
```

where:

```text
x  = forward position
y  = lateral position
vx = estimated longitudinal motion
vy = estimated lateral motion
```

Position is measured from the semantic matrix.

Velocity is estimated from the temporal evolution of position through the Kalman state estimator; it is not directly provided by the raw matrix.

### Detection vs track

A **detection** exists in one frame.

Example:

```text
car detected at x = 30 m, y = 2 m
```

A **track** is the temporal memory of an object across frames.

Example:

```text
Track 17
frame 100 -> measured
frame 101 -> measured
frame 102 -> temporarily coasted
frame 103 -> measured again
```

Data association decides which current detection is most compatible with each existing track.

### Data association

The implementation uses Hungarian/global assignment rather than matching objects independently in arbitrary order.

Association considers predicted position, uncertainty, radial/tangential gates, semantic object group, and class-change penalties.

Object identity is inferred from temporal/spatial consistency; the matrix itself does not contain ground-truth physical object IDs.

---

## 14. Live BEV viewer

`live_bev_viewer.py` is the visualization/integration wrapper for the online tracker.

It is different from the animation script:

```text
animate_target_speed_bev_cleaned.py
    -> processes recorded results and creates an MP4

live_bev_viewer.py
    -> updates the display immediately as each matrix arrives
```

### Replay recorded matrices as a live 10 Hz stream

```bash
python3 live_bev_viewer.py
```

The viewer reads one matrix, calls the causal tracker, renders the current result, waits according to the 10 Hz timing, then processes the next matrix.

### Test a short window

```bash
python3 live_bev_viewer.py \
    --start 1110 \
    --end 1190
```

### Process as fast as possible

```bash
python3 live_bev_viewer.py --no-realtime
```

### Headless mode

For a system without a graphical desktop:

```bash
python3 live_bev_viewer.py --headless
```

For a pure timing/processing test:

```bash
python3 live_bev_viewer.py \
    --headless \
    --no-realtime
```

### Use a different matrix directory

```bash
python3 live_bev_viewer.py \
    --matrix-dir /path/to/matrix
```

### Increase/decrease display scale

```bash
python3 live_bev_viewer.py --ppm 6
```

`--ppm` controls display pixels per meter only. It does not change the underlying 1 m BEV data resolution.

---

## 15. Using `LiveBEVProcessor` in another application

For integration into another Python process, use the wrapper directly:

```python
from live_bev_viewer import LiveBEVProcessor
import control_params as cp

processor = LiveBEVProcessor()

# Every time the perception system produces a new matrix:
result = processor.update(
    current_matrix,
    dt=cp.DT,
)

cleaned_matrix = result.cleaned_matrix
tracked_objects = result.objects
cleaned_target_speed_mph = result.cleaned_target_speed_mph
```

With IMU yaw rate:

```python
result = processor.update(
    current_matrix,
    dt=0.10,
    ego_yaw_rate=current_imu_yaw_rate,
)
```

The processor must remain persistent between frames.

---

## 16. NVIDIA Thor integration

The recommended deployment architecture is:

```text
                    NVIDIA THOR

Camera / sensor input
        |
        v
Perception / object detection
        |
        v
Distance + BEV position estimation
        |
        v
Current 120 x 80 uint8 matrix
        |
        v
OnlineTemporalCleaner.update()
        |
        +----------> cleaned matrix ----------> target speed / controller
        |
        +----------> continuous tracks -------> diagnostics / planning
        |
        +----------> visualization -----------> live BEV display
        |
        +----------> logger ------------------> recorded validation data
```

### Important design rule

The visualization must not be allowed to block the tracking/control path.

Recommended production design:

```text
perception -> tracker -> controller
                   |
                   +----> non-blocking viewer/logger
```

If the live viewer stops, tracking and control should continue.

### Integration pseudocode

```python
import time
import control_params as cp
from temporal_matrix_cleaner import OnlineTemporalCleaner
from target_speed import getTargetSpeed

tracker = OnlineTemporalCleaner()
last_time = None

while vehicle_system_running():
    raw_matrix, timestamp, yaw_rate = get_latest_perception_output()

    if last_time is None:
        dt = cp.DT
    else:
        dt = timestamp - last_time
    last_time = timestamp

    cleaned_matrix, objects = tracker.update(
        raw_matrix,
        dt=dt,
        ego_yaw_rate=yaw_rate,
    )

    target_speed = getTargetSpeed(
        cleaned_matrix,
        **cp.TARGET_SPEED_KW,
    )

    publish_cleaned_bev(cleaned_matrix)
    publish_tracks(objects)
    publish_target_speed(target_speed)
```

Before using measured variable `dt` in deployment, confirm that the tracker/API supports the expected variation and that timestamps are synchronized correctly. The development dataset itself is fixed at 0.10 s.

---

## 17. Streaming target-speed demonstration

`run_cleaned_target_speed.py` demonstrates the live causal call pattern using recorded matrices.

Run:

```bash
python3 run_cleaned_target_speed.py
```

Quiet summary mode:

```bash
python3 run_cleaned_target_speed.py --quiet
```

Process a selected frame range:

```bash
python3 run_cleaned_target_speed.py \
    --start 1110 \
    --end 1190
```

Write results to CSV:

```bash
python3 run_cleaned_target_speed.py \
    --csv streaming_target_speed.csv
```

Use a custom input directory:

```bash
python3 run_cleaned_target_speed.py \
    --matrix-dir /path/to/matrix
```

This script is useful for verifying the exact frame-by-frame deployment path without starting the graphical viewer.

---

## 18. Generate recorded comparison videos

### Offline comparison video

First generate offline outputs:

```bash
python3 run_temporal_cleaning.py \
    --mode offline \
    --out cleaned
```

Then:

```bash
python3 animate_target_speed_bev_cleaned.py \
    --cleaned-dir cleaned/matrix_cleaned \
    --tracks cleaned/tracks.npz \
    --out cleaned/bev_raw_vs_offline.mp4
```

### Causal comparison video

First generate causal outputs:

```bash
python3 run_temporal_cleaning.py \
    --mode causal \
    --out cleaned_causal
```

Then:

```bash
python3 animate_target_speed_bev_cleaned.py \
    --cleaned-dir cleaned_causal/matrix_cleaned \
    --tracks cleaned_causal/tracks.npz \
    --out cleaned_causal/bev_raw_vs_causal.mp4
```

### Selected video range

```bash
python3 animate_target_speed_bev_cleaned.py \
    --start 1110 \
    --end 1190 \
    --cleaned-dir cleaned_causal/matrix_cleaned \
    --tracks cleaned_causal/tracks.npz \
    --out cleaned_causal/test_1110_1190.mp4
```

### Provenance visualization

```bash
python3 animate_target_speed_bev_cleaned.py \
    --start 1110 \
    --end 1190 \
    --show-provenance \
    --out cleaned/worst_case_1110_1190.mp4
```

---

## 19. Output files

A run of `run_temporal_cleaning.py` can produce files such as:

```text
cleaned/
├── matrix_cleaned/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── tracks.npz
├── egomotion.npz
├── config.json
├── manifest.json
└── ...
```

### `matrix_cleaned/`

Rasterized cleaned matrices using the original:

```text
120 x 80
1 m / cell
uint8 semantic representation
```

This preserves compatibility with the downstream target-speed function.

### `tracks.npz`

Continuous tracked-object sidecar data. Depending on the current code version, fields include information such as:

```text
frame
track ID
semantic class
continuous forward position
continuous lateral position
velocity components
observed/coasted/interpolated/pass-through provenance
```

Use the sidecar when continuous sub-cell track information is needed.

### `egomotion.npz`

Estimated ego-motion quantities used by the tracker/analysis.

### `config.json`

Configuration used for the cleaning run.

### `manifest.json`

Run metadata such as mode, number of frames, frame rate, duration, geometry, and source information.

---

## 20. Evaluate raw vs cleaned output

Run the standard comparison:

```bash
python3 compare_raw_cleaned.py \
    --json cleaned/report.json
```

The comparison script can evaluate raw, offline, and causal output using the same reference-association logic.

### Residual failure audit

```bash
python3 compare_raw_cleaned.py \
    --audit-failures \
    --top 15
```

### Resolution study

```bash
python3 compare_raw_cleaned.py \
    --resolution-study
```

Important metrics include:

```text
dropout fraction
short dropout events
trajectory jumps
second-difference smoothness
track fragmentation
class instability
duplicate stop hypotheses
target-speed changes
one-frame braking episodes
safety-audit failures
```

---

## 21. Causality validation

A real online tracker must satisfy the following property:

> Output at frame `k` must not change just because frames after `k` are available.

A useful validation procedure is:

```text
Run A: process frames 0 ... 300
Run B: process frames 0 ... 500

Compare outputs 0 ... 300.
They must be identical.
```

The current causal implementation was developed specifically to satisfy this prefix-invariance requirement.

Another important validation is:

```text
batch causal sequence
==
calling OnlineTemporalCleaner.update() one frame at a time
```

If these ever stop matching after future code changes, investigate before deploying the modified version.

---

## 22. Performance / real-time requirement

At 10 Hz, each frame has a nominal processing budget of:

```text
100 ms
```

The tracking computation should be benchmarked on the actual deployment hardware, not only on a development workstation.

Use:

```bash
python3 live_bev_viewer.py \
    --headless \
    --no-realtime
```

or:

```bash
python3 run_cleaned_target_speed.py --quiet
```

for local timing tests.

For Thor deployment, separately benchmark:

```text
perception latency
tracker latency
target-speed latency
inter-process communication latency
visualization latency
end-to-end frame age
```

Visualization/encoding time should not be counted as tracker compute time when evaluating the control deadline.

---

## 23. Safety behavior

The cleaner includes a safety pass-through intended to prevent a newly observed control-relevant object from being hidden merely because it has not yet accumulated enough temporal history.

Important principles:

- a new physical track originates from an actual current detection;
- causal prediction may bridge short missed detections for an existing track;
- the online system does not create arbitrary new physical objects from prediction alone;
- the cleaned result should be audited for cases where a control-relevant detection moves materially farther away;
- the original raw matrix should remain available for logging and validation.

Do not remove or weaken the safety pass-through solely to make the visualization smoother.

---

## 24. Offline vs causal: which one should I use?

Use this rule:

| Goal | Mode |
|---|---|
| Best trajectory reconstruction on recorded data | Offline |
| Research analysis | Offline |
| Produce offline comparison figures/video | Offline |
| Validate deployment behavior on recorded route | Causal |
| Real-time vehicle deployment | Causal / `OnlineTemporalCleaner.update()` |
| Live BEV display | Causal |
| Longitudinal controller input | Causal |

Never use RTS-smoothed/future-informed offline states in a real-time controller.

---

## 25. Recommended deployment sequence

Do not go directly from a laptop experiment to active vehicle control.

Recommended sequence:

```text
1. Replay recorded matrices offline
2. Validate causal output
3. Run frame-by-frame causal replay
4. Integrate tracker after the live BEV estimator
5. Display live BEV only
6. Log raw + cleaned BEV + tracks + timing
7. Run in shadow mode with target speed calculated but not commanded
8. Compare raw and cleaned target-speed behavior
9. Benchmark on NVIDIA Thor
10. Verify reset/startup/dropout behavior
11. Only then connect the validated causal output to active control
```

---

## 26. Resetting the tracker

The online tracker contains temporal state.

Create/reset it when starting a new independent drive/session:

```python
tracker = OnlineTemporalCleaner()
```

Do not reuse track history from an unrelated previous route unless that behavior is intentionally designed and tested.

If the live wrapper exposes a reset method, use that method at route/session reset. Otherwise instantiate a new tracker between sessions.

---

## 27. Troubleshooting

### `ModuleNotFoundError`

Install dependencies:

```bash
python3 -m pip install numpy scipy matplotlib opencv-python
```

### `matrix/` not found

Make sure the data directory exists:

```text
bef-tracking-kf/
├── matrix/
│   ├── 000000.npy
│   └── ...
└── run_temporal_cleaning.py
```

Or provide it explicitly:

```bash
python3 run_temporal_cleaning.py \
    --matrix-dir /absolute/path/to/matrix
```

### Wrong matrix shape

Expected:

```text
(120, 80)
```

Check:

```python
print(matrix.shape)
```

### Wrong matrix dtype

Expected:

```text
uint8
```

Check:

```python
print(matrix.dtype)
```

### Tracking seems to restart every frame

Make sure the tracker is instantiated once:

```python
tracker = OnlineTemporalCleaner()
```

outside the frame-processing loop.

### OpenCV window does not open

The system may be headless.

Use:

```bash
python3 live_bev_viewer.py --headless
```

For a remote/headless Thor deployment, publish the visualization image to a separate display workstation rather than requiring a local GUI.

### Live viewer is slow

Test tracker computation without display pacing:

```bash
python3 live_bev_viewer.py \
    --headless \
    --no-realtime
```

If this is fast but the GUI is slow, the bottleneck is visualization rather than tracking.

### Timing is incorrect

For the development dataset:

```text
FPS = 10
dt = 0.10 s
```

Check `control_params.py` rather than introducing duplicate timing constants.

### Git repository tries to add huge data/video folders

Add generated artifacts to `.gitignore` before committing.

Recommended patterns:

```gitignore
__pycache__/
*.pyc
.venv/

matrix/
matrix_cleaned/
cleaned/
cleaned_causal/

*.mp4
*.npz

.DS_Store
```

Run:

```bash
git status
```

before every large commit.

---

## 28. Recommended `.gitignore`

A typical repository should not commit recorded matrices, generated cleaned sequences, or large videos into standard Git history.

Example:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Dataset and generated outputs
matrix/
matrix_cleaned/
cleaned/
cleaned_causal/

# Generated media / sidecars
*.mp4
*.npz

# OS/editor
.DS_Store
.vscode/
.idea/
```

If the team intentionally wants to version a small sample dataset, add a dedicated small folder such as `examples/sample_matrix/` and explicitly un-ignore only that folder.

---

## 29. Suggested `requirements.txt`

Create `requirements.txt` with:

```text
numpy
scipy
matplotlib
opencv-python
```

Then installation becomes:

```bash
python3 -m pip install -r requirements.txt
```

For production deployment, pin package versions after testing the exact Thor software environment.

---

## 30. Example integration module

A minimal reusable integration pattern is:

```python
import control_params as cp
from temporal_matrix_cleaner import OnlineTemporalCleaner
from target_speed import getTargetSpeed


class BEVTrackingPipeline:
    def __init__(self):
        self.tracker = OnlineTemporalCleaner()

    def update(self, raw_matrix, ego_yaw_rate=None):
        cleaned_matrix, objects = self.tracker.update(
            raw_matrix,
            dt=cp.DT,
            ego_yaw_rate=ego_yaw_rate,
        )

        target_speed = getTargetSpeed(
            cleaned_matrix,
            **cp.TARGET_SPEED_KW,
        )

        return cleaned_matrix, objects, target_speed
```

Usage:

```python
pipeline = BEVTrackingPipeline()

while running:
    raw_matrix = receive_current_bev()
    yaw_rate = receive_current_yaw_rate()

    cleaned, objects, target_speed = pipeline.update(
        raw_matrix,
        ego_yaw_rate=yaw_rate,
    )

    publish_cleaned_bev(cleaned)
    publish_target_speed(target_speed)
```

---

## 31. Development-data results

The project report for the development sequence showed that causal tracking remained substantially better than raw input while being future-information-free.

Representative reported metrics included:

| Metric | Raw | Offline | Causal |
|---|---:|---:|---:|
| Dropout fraction | 0.2296 | 0.0372 | 0.0494 |
| Short 1–3 frame dropouts | 1732 | 73 | 149 |
| Unexplained longitudinal jumps > 3 m | 890 | 22 | 197 |
| Unexplained lateral jumps > 2 m | 668 | 28 | 130 |
| Mean second-difference magnitude | 2.033 | 0.710 | 0.899 |
| Target-speed changes > 10 mph | 30 | 13 | 20 |
| One-frame braking episodes | 24 | 11 | 16 |

These values describe the development dataset and should not be treated as guaranteed performance on a new route, vehicle, sensor configuration, or software version.

Always rerun validation after changing perception, geometry, timing, tracker parameters, vehicle platform, or sensor synchronization.

---

## 32. Known limitations

The tracker improves temporal consistency but does not create new sensor information.

Current limitations include:

- absolute range/depth bias cannot be fully corrected without an independent distance source;
- object identity is inferred rather than provided by ground-truth IDs;
- sufficiently long detection gaps can break track identity;
- close objects can create association ambiguity;
- causal filtering cannot use future observations to retroactively repair a past outlier;
- the final 1 m rasterized matrix still contains quantization;
- the continuous sidecar track representation is smoother than the 1 m control raster;
- deployment performance must be benchmarked on the actual NVIDIA Thor system;
- live sensor timestamps, yaw-rate units, coordinate conventions, and synchronization must be verified during integration.

---

## 33. Important things not to change casually

Before modifying any of the following, rerun the complete validation suite:

```text
BEV geometry
Ego origin
Matrix class IDs
DT / FPS
Kalman process noise
measurement covariance
association gates
class grouping
coasting limits
safety pass-through
stop-sign handling
target-speed interface
```

Do not tune the tracker only to make one video look smoother.

Changes should be justified by full-sequence metrics and safety checks.

---

## 34. First-time user checklist

Before considering the repository correctly installed:

```text
[ ] Repository cloned
[ ] Python environment created
[ ] Dependencies installed
[ ] Input matrices are 120 x 80 uint8
[ ] Matrix geometry is understood
[ ] Class IDs match the documented mapping
[ ] control_params.py reports 10 Hz / 0.10 s
[ ] Offline cleaner runs
[ ] Causal cleaner runs
[ ] run_cleaned_target_speed.py runs
[ ] Live BEV viewer runs or headless mode works
[ ] OnlineTemporalCleaner is created once, not once per frame
[ ] Causal output uses no future information
[ ] Raw and cleaned data are logged during integration
[ ] Thor latency is benchmarked
[ ] Shadow-mode validation is completed before active control
```

---

## 35. Common commands cheat sheet

### Clone

```bash
git clone https://github.com/Labid52/bef-tracking-kf.git
cd bef-tracking-kf
```

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib opencv-python
```

### Offline clean

```bash
python3 run_temporal_cleaning.py --mode offline --out cleaned
```

### Causal clean

```bash
python3 run_temporal_cleaning.py --mode causal --out cleaned_causal
```

### Streaming target-speed demo

```bash
python3 run_cleaned_target_speed.py
```

### Live BEV replay

```bash
python3 live_bev_viewer.py
```

### Live BEV headless benchmark

```bash
python3 live_bev_viewer.py --headless --no-realtime
```

### Offline video

```bash
python3 animate_target_speed_bev_cleaned.py \
    --cleaned-dir cleaned/matrix_cleaned \
    --tracks cleaned/tracks.npz \
    --out cleaned/bev_raw_vs_offline.mp4
```

### Causal video

```bash
python3 animate_target_speed_bev_cleaned.py \
    --cleaned-dir cleaned_causal/matrix_cleaned \
    --tracks cleaned_causal/tracks.npz \
    --out cleaned_causal/bev_raw_vs_causal.mp4
```

### Evaluation

```bash
python3 compare_raw_cleaned.py --json cleaned/report.json
```

### Residual audit

```bash
python3 compare_raw_cleaned.py --audit-failures --top 15
```

### Resolution study

```bash
python3 compare_raw_cleaned.py --resolution-study
```

---

## 36. Git workflow

Check files first:

```bash
git status
```

Add source/documentation files:

```bash
git add .
```

Check again before committing:

```bash
git status
```

Make sure large `matrix/`, `cleaned/`, `cleaned_causal/`, `.mp4`, and `.npz` artifacts are not accidentally staged.

Commit:

```bash
git commit -m "Update BEV temporal tracking"
```

Push:

```bash
git push origin main
```

---

## 37. License

This repository is distributed under the MIT License.

See the existing `LICENSE` file in the repository for the full license text.

---

## 38. Summary for integrators

If you only need the deployment interface, remember these four points:

1. Upstream must provide one current `120 x 80 uint8` semantic matrix.
2. Instantiate `OnlineTemporalCleaner` once.
3. Call `update(current_matrix, dt=0.10, ...)` once per new frame.
4. Use the returned cleaned matrix/tracks immediately; do not use offline RTS-smoothed results in real-time control.

Minimal deployment pattern:

```python
from temporal_matrix_cleaner import OnlineTemporalCleaner

tracker = OnlineTemporalCleaner()

while running:
    raw_matrix = get_current_matrix()

    cleaned_matrix, objects = tracker.update(
        raw_matrix,
        dt=0.10,
    )

    use_current_cleaned_bev(cleaned_matrix, objects)
```

That is the intended real-time interface of this repository.
