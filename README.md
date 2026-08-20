# BEV Temporal Tracking with Kalman Filtering

This repository stabilizes a sparse Bird's-Eye-View (BEV) semantic matrix before it is used for longitudinal control.

The pipeline is:

```text
Object detection + distance/BEV estimation
        ↓
Raw 120×80 semantic matrix
        ↓
OnlineTemporalCleaner
(Kalman prediction + data association + correction)
        ↓
Stabilized BEV
        ├── Live BEV display
        └── Target-speed calculation / controller
```

The deployment/online mode is **causal**: each output uses only the current frame and tracker state from previous frames. Future frames are not used.

---

## Input

Each input is a NumPy semantic matrix:

```python
shape = (120, 80)
dtype = np.uint8
```

Geometry:

- Ego reference: `(row=80, col=40)`
- Resolution: `1 m/cell`
- Update rate: `10 Hz`
- `dt = 0.10 s`
- Forward distance: `x = 80 - row`
- Lateral distance: `y = col - 40`

Each nonzero cell is treated as a **point detection** at the estimated object/front-face location, not as the full physical footprint of the object.

Main class IDs:

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

---

## Main Files

| File | Purpose | When to use |
|---|---|---|
| `temporal_matrix_cleaner.py` | Main Kalman tracking, data association, offline and causal tracking | Core library; imported by other scripts |
| `control_params.py` | Shared timing and controller parameters | Keep `DT=0.10` here |
| `target_speed.py` | Target-speed calculation from BEV | Used after the cleaned matrix is produced |
| `run_temporal_cleaning.py` | Process a recorded matrix sequence | Generate offline or causal cleaned datasets |
| `run_cleaned_target_speed.py` | Simple frame-by-frame causal/streaming example | Check the online API |
| `live_bev_viewer.py` | Real-time BEV visualization | Use to watch the causal BEV update live |
| `animate_target_speed_bev_cleaned.py` | Create comparison MP4 videos | Recorded-data visualization only |
| `compare_raw_cleaned.py` | Quantitative evaluation/audit | Compare raw, offline, and causal results |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Labid52/bef-tracking-kf.git
cd bef-tracking-kf
```

Install the required Python packages:

```bash
python3 -m pip install numpy scipy matplotlib opencv-python
```

`ffmpeg` is recommended if you want to generate MP4 videos.

---

## Recorded Matrix Data

For recorded-data scripts, place the matrices in:

```text
matrix/
├── 000000.npy
├── 000001.npy
├── 000002.npy
└── ...
```

The raw `matrix/` directory should remain unchanged.

---

## 1. Run Offline Cleaning

Offline mode uses future observations, RTS smoothing, and tracklet stitching. Use it for analysis and best-quality reconstructed trajectories, **not for live vehicle control**.

```bash
python3 run_temporal_cleaning.py
```

Output:

```text
cleaned/
```

---

## 2. Run Causal / Online Cleaning on Recorded Data

This processes the recorded route in causal order using only past + current information:

```bash
python3 run_temporal_cleaning.py \
    --mode causal \
    --out cleaned_causal
```

Output:

```text
cleaned_causal/
```

Use this mode when evaluating behavior that is representative of real-time deployment.

---

## 3. Watch the BEV Update in Real Time

To replay recorded matrices one at a time at 10 Hz and watch the live causal BEV:

```bash
python3 live_bev_viewer.py
```

For a short section:

```bash
python3 live_bev_viewer.py --start 1110 --end 1190
```

For processing without a GUI:

```bash
python3 live_bev_viewer.py --headless --no-realtime
```

`live_bev_viewer.py` is different from the animation script:

- `live_bev_viewer.py` → displays the current causal output as frames arrive.
- `animate_target_speed_bev_cleaned.py` → creates an MP4 from recorded results.

---

## 4. Use the Tracker in a Live Perception Pipeline

Create the tracker **once**, then call `update()` for every new BEV matrix:

```python
from temporal_matrix_cleaner import OnlineTemporalCleaner

tracker = OnlineTemporalCleaner()

# Called once for each new matrix from perception
cleaned_matrix, objects = tracker.update(
    current_matrix,
    dt=0.10,
)
```

Do **not** create a new tracker every frame. The tracker object stores the previous Kalman states, velocities, IDs, uncertainty, and missed-detection history.

If vehicle/IMU yaw rate is available:

```python
cleaned_matrix, objects = tracker.update(
    current_matrix,
    dt=0.10,
    ego_yaw_rate=current_yaw_rate,
)
```

Recommended live architecture:

```text
Perception
   ↓
Current raw BEV matrix
   ↓
OnlineTemporalCleaner.update()
   ↓
Current stabilized BEV
   ├── live_bev_viewer / display
   └── getTargetSpeed() → longitudinal controller
```

The visualization should be kept separate from the control loop so control continues even if the display is disabled.

---

## 5. Generate Comparison Videos

Offline:

```bash
python3 animate_target_speed_bev_cleaned.py \
    --out cleaned/bev_raw_vs_offline.mp4
```

Causal:

```bash
python3 animate_target_speed_bev_cleaned.py \
    --cleaned-dir cleaned_causal/matrix_cleaned \
    --tracks cleaned_causal/tracks.npz \
    --out cleaned_causal/bev_raw_vs_causal.mp4
```

---

## 6. Evaluate the Results

Run the comparison report:

```bash
python3 compare_raw_cleaned.py --json cleaned/report.json
```

Audit remaining severe events:

```bash
python3 compare_raw_cleaned.py --audit-failures --top 15
```

Useful metrics include dropout rate, trajectory jumps, fragmentation, duplicate tracks, target-speed jumps, and safety-audit failures.

---

## Offline vs Online

| Feature | Offline | Online/Causal |
|---|---:|---:|
| Kalman prediction/correction | Yes | Yes |
| Data association | Yes | Yes |
| Short dropout handling | Yes | Yes |
| Future frames | Yes | **No** |
| RTS backward smoothing | Yes | **No** |
| Future tracklet stitching | Yes | **No** |
| Suitable for live deployment | No | **Yes** |

The causal implementation has been validated so that adding future frames does not change earlier outputs. It can therefore be used as the basis for real-time integration.

---

## Important Notes

- Keep the tracker instance alive across frames.
- Use `dt = 0.10 s` for the current 10 Hz BEV stream.
- Do not use offline-smoothed output in a live controller.
- Tracking improves temporal consistency but does not create new sensor information or correct unknown absolute range bias.
- Benchmark latency again on the final NVIDIA Thor deployment hardware.
- Perform shadow-mode/replay validation before allowing the cleaned BEV to affect vehicle actuation.

---

## License

MIT License. See `LICENSE`.
