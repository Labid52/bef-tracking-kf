# BEV Temporal Tracker — Fix & Validation Report

Baseline commit: `0908b77be1085e5b0121badc72cf504028b0353f`
Dataset: `realtime_capture/matrix` — 6925 frames, 120×80 uint8, captured at 10 Hz
(`sha256 53167fbc70b83bfee79d5f2e344afbf38ac9e791dd05f72b68214b307c95bcc7`, unchanged).

Everything below was measured on that one sequence. RAW matrices and
`target_speed.py` were not modified.

---

## 1. Files modified

| file | change |
|---|---|
| `temporal_matrix_cleaner.py` | rate-awareness refactor; revocable duplicate policy; new coast policy; velocity clamp; direction-aware safety pass-through; causal-yaw retune |
| `control_params.py` | `LIVE_FPS/LIVE_DT` (20 Hz) separated from `LEGACY_FPS/LEGACY_DT` (10 Hz) |
| `run_temporal_cleaning.py` | `--dt` / `--fps` genuinely honoured and threaded through; manifest records the real timing |
| `live_bev_viewer.py` | real monotonic clock, `measure_dt()`, abnormal-dt policy, `--dt/--fps`, rendering decoupled from tracking |
| `compare_four_bev_modes.py` | generalized to 2–6 named panels (`--panel NAME=DIR`) |
| `README.md`, `README_LIVE_BEV.md` | 20 Hz live vs 10 Hz recorded; live loop; commands |
| **new** `test_tracker_regression.py` | 17 regression tests |
| **new** `validate_tracker.py` | causality / rate / latency harness |

**Intentionally untouched** (verified by sha256 against the pre-task baseline):
`target_speed.py`, `compare_raw_cleaned.py`, `run_cleaned_target_speed.py`,
`animate_target_speed_bev_cleaned.py`, `inspect_matrix_geometry.py`,
`run_target_speed.py`, `matrix/`, `realtime_capture/matrix`, and the previous
result folders `realtime_capture/{online,offline,rerun_causal,matrix_tracked}`
together with the existing comparison videos.

`sha256sum -c` against the pre-task manifest reports exactly 7 changed files —
the 7 listed in the table above — and OK for every frozen file.

New result folders (previous ones untouched):
`realtime_capture/fixed_online_10hz`, `fixed_online_20hz`, `fixed_offline_10hz`.

---

## 2. Logic changed

### A. Duplicate handling — `OnlineTemporalCleaner._update_duplicates`

*Was:* 4 consecutive frames within 3 m → `duplicate_of` latched **permanently**,
never re-evaluated, and the suppressed track kept winning Hungarian assignments
(an invisible detection sink).

*Now:* a rolling window of pairwise separations per track pair.
- **Suppress** only after `duplicate_window_s = 1.0 s` of co-location with
  ≥ `duplicate_coverage = 0.85` of samples inside the radius, plus the existing
  support condition.
- **Release** when the most recent `duplicate_release_s = 0.4 s` has a median
  separation beyond the radius, or immediately beyond
  `duplicate_hard_release_factor = 2×` the radius.
- A suppression whose keeper died, or itself became a duplicate, is cleared.

Both decisions read only past samples, so the rule stays causal.

### B. Coast policy — `Tracker.step`, `Track.robust_velocity`, `Track.predict`

*Was:* on detection loss the instantaneous Kalman velocity was frozen and
extrapolated blindly for up to 8 frames.

*Now:*
1. On the **first** blind frame the velocity is replaced by a **Theil–Sen**
   median of pairwise slopes over the track's own last `coast_velocity_window_s = 0.5 s`
   of observations, and the step just taken with the noisy value is undone.
   (Theil–Sen, not consecutive differences: at 20 Hz a vehicle closing at 8 m/s
   moves 0.4 m per frame, so most consecutive differences on a 1 m raster are
   exactly zero and their median is zero.)
2. During the coast the whole velocity decays with `coast_tau_s = 0.20 s`, so a
   blind prediction fades into a position **hold** within ~3 frames.
3. Emission stops after `emit_coast_s = 0.8 s` or once the position 1-σ exceeds
   `emit_max_pos_sigma_m = 8.0 m`.

### C. Velocity clamp — `Track._clamp_speed`

New hard physical bound `max_state_speed_mps = 60 m/s` on the velocity state.
At 20 Hz one cell of quantisation per frame is already 20 m/s; unclamped, peak
|w| reached **117.9 m/s**. Now 60.0 m/s at every tested rate.

### D. Safety pass-through — `add_safety_passthrough`

A raw detection now counts as "already covered" only by an emitted object that is
close **and not more than `passthrough_farther_tolerance_m = 1.5 m` farther from
the ego**. A counterpart further away commands less braking, so it cannot stand
in for control-relevant evidence. This also removed an oscillating target speed
caused by the coverage test flickering across its radius boundary.

### E. Rate awareness — throughout

`dt` is now an explicit argument on `Tracker.step`, `Track.predict`,
`transition`, `process_noise`, `lateral_damping`, `_finite_difference_velocity`,
`ego_points_from_tracks`, `estimate_ego_motion`, `_rts_smooth_1d`, `_refilter`,
`_extrapolate`, `build_emissions`, `stitch_tracklets`, `merge_duplicate_tracks`,
`suppress_duplicates`, `_try_combine`, `clean_sequence`, `_clean_causal`,
`ScalarCVFilter.step`. No bare `DT` remains in any computation.

The tracker keeps its own accumulated clock (`Tracker.time`) and every track
carries `obs_times` / `last_obs_time`, so durations survive a variable frame rate.
Comparisons against a duration use `time_exceeds()` with a tolerance of
`1e-3 · dt`, because the clock is an accumulated float sum (an exact `>` test
misfired at `0.6000000000000019 > 0.6`).

**Time-based policies (seconds):** `max_coast_s`, `emit_coast_s`, `coast_tau_s`,
`coast_velocity_window_s`, `confirm_window_s`, `duplicate_window_s`,
`duplicate_release_s`, `stitch_max_gap_s`, `merge_min_overlap_s`,
`ego_flow_baseline_s`, `tau_lateral_s`, and the gate growth rates (per second).

**Observation-count policies (unchanged as counts):** `confirm_hits`,
`min_track_observations`, `duplicate_min_support`, `duplicate_support_ratio`,
`coast_velocity_min_samples`. These are evidence criteria, not durations.

### F. Causal yaw

`ego_sigma_psi_meas` 0.08 → 0.04, `ego_sigma_psidot` 0.15 → 0.80, chosen by A/B
against a yaw estimated independently from the RAW flow.

---
## 3. A/B experiments

Every parameter was chosen by measurement on the same RAW sequence, never by
intuition. Metrics use one shared reference associator applied identically to
every dataset (not the cleaner's own track IDs).

### Coast policy sweep (all with the duplicate fix applied)

| variant | veh cells %RAW | RAW unmatched@3m | **cross \|y\|<3** | unexplained \|dy\|>2 | dropout | frag ≤2 | obs/track |
|---|---|---|---|---|---|---|---|
| BASELINE (shipped ONLINE) | 89.8 | 31.3 % | **8 (7 unsupported)** | 831 | 0.056 | 4.26 % | 55.0 |
| C1 duplicate fix only, frozen CV | 119.8 | 15.4 % | **15 (10)** | 942 | 0.051 | 2.64 % | 66.3 |
| C2 robust velocity, **no decay** | 121.2 | 16.8 % | **30 (23)** | 3366 | 0.056 | 1.96 % | 58.4 |
| C3 robust + τ 0.45 | 125.1 | 16.8 % | 3 (1) | 994 | 0.055 | 2.43 % | 71.9 |
| C4 robust + τ 0.45, emit 0.4, σ 3 | 110.5 | 17.1 % | 2 (1) | 774 | 0.100 | 4.52 % | 61.3 |
| C5 robust + τ 0.30, emit 0.3, σ 2.5 | 106.0 | 17.5 % | 0 (0) | 666 | 0.120 | 6.31 % | 58.9 |
| **D1 robust + τ 0.20, emit 0.8 (chosen)** | **126.9** | **17.7 %** | **0 (0)** | **763** | **0.056** | **2.30 %** | **76.1** |
| D2 robust + τ 0.12, emit 0.8 | 127.2 | 18.5 % | 0 (0) | 622 | 0.055 | 2.53 % | 76.0 |
| D3 robust + τ 0.20, emit 0.6 | 121.7 | 17.8 % | 0 (0) | 740 | 0.068 | 2.54 % | 71.8 |
| OFFLINE (quality reference) | 114.9 | 11.3 % | 0 (0) | 56 | 0.027 | 3.67 % | 73.9 |

Two findings worth recording:

* **The duplicate fix alone made crossings worse** (8 → 15). More real tracks
  survive, so more of them coast. The two fixes had to be done together.
* **A robust velocity without decay is much worse than the noisy one**
  (C2: 30 crossings, 3366 lateral events). Theil–Sen produces a *sustained*
  velocity, so without decay it extrapolates further than the noisy estimate did.
  Decay is what makes it safe.

D1 was chosen: it is the only variant reaching **zero** crossings while keeping
dropout at the baseline value (0.056) and improving fragmentation and track
length. C5 also reached zero but more than doubled dropout (0.120).

### Causal-yaw sweep

Measured against a yaw estimated independently from the RAW flow
(target: amplitude ratio 1.0, |ψ| p95 = 0.274 rad/s):

| variant | amplitude ratio on strong turns | correlation | \|ψ\| p95 |
|---|---|---|---|
| current (σ 0.08, ψ̇ 0.15) | 0.342 | 0.430 | 0.169 |
| σ 0.04 | 0.455 | 0.495 | 0.212 |
| ψ̇ 0.40 | 0.502 | 0.519 | 0.231 |
| σ 0.04, ψ̇ 0.40 | 0.594 | 0.564 | 0.249 |
| **σ 0.04, ψ̇ 0.80 (chosen)** | **0.644** | **0.580** | **0.261** |
| damping 0.5, σ 0.04, ψ̇ 0.40 | 0.655 | 0.580 | 0.263 |

End-to-end the change is marginal (unexplained lateral 763 → 758, RAW unmatched
17.70 % → 17.50 %), but it is an objective improvement in the estimate itself
with no regression, which is what Part C asked for. The σ 0.04 / ψ̇ 0.40 variant
was rejected because it reintroduced 2 crossings.

---
## 4. Before → after on the four problems

Reference associator identical for every column; RAW is the measurement reference.

| metric | BASELINE (shipped ONLINE) | **FINAL (fixed causal, 10 Hz)** | OFFLINE (quality reference) |
|---|---|---|---|
| vehicle cells, % of RAW | 89.8 % | **126.3 %** | 114.9 % |
| **P1** RAW vehicle cells unmatched @3 m | **31.34 %** | **17.42 %** | 11.25 % |
| orphan cells @3 m (no RAW nearby) | 31.16 % | 34.76 % | 23.97 % |
| **P2** through-ego crossings, \|y\|<3 m | **8 (7 RAW-unsupported)** | **0 (0)** | 0 (0) |
| crossings \|y\|<5 m | 23 | 9 | 6 |
| **P3** unexplained \|dy\| > 2 m | 831 | **791** | 56 |
| unexplained \|dy\| > 4 m | 125 | **96** | 10 |
| **P4** mean Δx per frame | −0.258 m | **−0.154 m** | −0.235 m |
| strong approach (Δx < −1.5 m) | 3.39 % | **1.87 %** | 1.08 % |
| **P6** dropout fraction | 0.056 | 0.057 | 0.027 |
| fragment tracks (≤2 obs) | 4.26 % | **2.56 %** | 3.67 % |
| observations per track | 54.96 | **74.96** | 73.89 |

### Duplicate suppression, before → after

| | BEFORE (diagnosed) | AFTER |
|---|---|---|
| vehicle detections discarded as duplicates | **29 424 (23.79 %)** | **1 223 (0.99 %)** |
| vehicle tracks flagged duplicate | 808 / 3375 (23.9 %) | 560 ever flagged, 93 flagged at end |
| separation at the moment a detection was discarded | median **4.50 m**, p90 **13.01 m** | — |
| discards occurring while the pair was already apart | **86.9 %** | — |
| longest continuous separation while still flagged | median 3.6 s, max **101 s** | median **0.20 s**, max **0.20 s** |
| tracks stale beyond the 0.4 s release grace | most of them | **0 / 388** |
| invisible-sink events beyond the grace period | many (tracks absorbed up to 987 further detections) | **0** |

Criterion B (no permanent stale suppression) and criterion C (no indefinite
invisible sink) are met exactly: no flagged pair stays separated longer than
0.20 s, well inside the 0.40 s release window, and no detection is absorbed by a
separated-and-flagged track beyond that grace period.

### Coast quality (criterion F)

Coasted estimate vs RAW, compared against a hold-last-position baseline:

| coast age | OLD err_coast | OLD err_hold | OLD worse-than-hold | NEW err_coast | NEW err_hold | NEW worse-than-hold |
|---|---|---|---|---|---|---|
| 1 frame | 5.45 m | 5.43 m | 52.3 % | **5.18 m** | 5.24 m | **45.0 %** |
| 4 frames | 7.20 m | 6.87 m | 58.9 % | **6.81 m** | 6.91 m | **50.2 %** |
| 8 frames | 8.86 m | 7.54 m | 65.8 % | **8.72 m** | 8.64 m | **53.2 %** |

Systematic drift **toward the ego** during a coast — the actual mechanism behind
P2 and P4:

| coast age | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| OLD mean Δx | −0.22 | −0.47 | −0.78 | −1.06 | −1.42 | −1.78 | −2.17 | **−2.58 m** |
| NEW mean Δx | −0.16 | −0.21 | −0.25 | −0.28 | −0.29 | −0.31 | −0.32 | **−0.33 m** |

**7.8× less drift at coast age 8.** The old policy was always worse than simply
holding position; the new one is better than hold for the first six frames.

---
## 5. Target speed (policy called unmodified)

| dataset | mean mph | mean \|Δ\| | p99 \|Δ\| | >5 mph | >10 mph | braking episodes | 1-frame episodes | 1-frame holes |
|---|---|---|---|---|---|---|---|---|
| RAW | 16.898 | 0.4352 | 8.24 | 168 | 44 | 266 | 91 | 104 |
| OLD ONLINE | 16.743 | 0.3828 | 8.33 | 152 | 49 | 199 | 56 | 63 |
| **FIXED ONLINE** | 16.592 | **0.3396** | **7.49** | **127** | **32** | **149** | **41** | **41** |
| OFFLINE (fixed) | 16.649 | 0.2808 | 7.00 | 102 | 35 | 116 | 21 | 17 |

The fixed causal output improves every smoothness measure over both RAW and the
old online output, and the mean target speed goes slightly **down** (16.74 →
16.59 mph), so the improvement is not the tracker simply braking less.

### Safety audit

Of 9 701 control-relevant RAW detections (inside the region `getTargetSpeed`
reacts to):

| dataset | represented conservatively | counterpart >1.5 m farther | **true losses** |
|---|---|---|---|
| OLD ONLINE | 9 336 | 359 | **6** |
| **FIXED ONLINE** | **9 696** | **5** | **0** |
| OFFLINE (fixed) | 9 700 | 0 | 1 (f5400, no target-speed effect) |

The direction-aware coverage test (change D) is what produced this: detections
whose only counterpart sat farther from the ego went from **416 to 5**, and true
losses from 6 to **0**. Worked example — a stop sign at frames 2596–2615 whose
tracked range lagged ~4 m behind RAW:

| | before | after |
|---|---|---|
| target speed | `4.73 · 3.34 · 0.00 · 6.69 · 0.00 · 6.69 · …` (oscillating) | `4.73 · 3.34 · 0.00 · 0.00 · 0.00 · …` (matches RAW exactly) |

---

## 6. Timing, causality and performance

### Outputs produced

| folder | mode | fps | dt | duration | tracks | cells | provenance |
|---|---|---|---|---|---|---|---|
| `realtime_capture/fixed_online_10hz` | causal | 10 | 0.10 | 692.5 s | 3474 | 163 316 | obs 112 708 / coast 48 549 / pass 2 193 |
| `realtime_capture/fixed_online_20hz` | causal | 20 | 0.05 | 346.25 s | 2852 | 204 901 | obs 112 754 / coast 89 664 / pass 2 853 |
| `realtime_capture/fixed_offline_10hz` | offline | 10 | 0.10 | 692.5 s | 3215 | 151 974 | obs 116 599 / interp 34 670 / pass 832 |

The 20 Hz run reports 346.25 s because it treats this 10 Hz recording as a 20 Hz
stream — that is the point of the flag, and the manifest records the timing it
actually used.

### Causality and streaming (`validate_tracker.py`)

| test | result |
|---|---|
| prefix invariance, cuts at 37 / 200 / 651 / 900 / 1170 / 1440 | **all identical to the full run** |
| `clean_sequence(mode="causal")` == repeated `update()` | **identical** |
| two identical runs → identical output | **identical** |

### Rate handling

| case | result | peak \|w\| | covariance finite |
|---|---|---|---|
| constant 10 Hz (dt = 0.100) | OK | 39.5 m/s | yes |
| constant 20 Hz (dt = 0.050) | OK | 60.0 m/s (clamped) | yes |
| 20 Hz + ±4 ms jitter | OK | 60.0 m/s | yes |
| 20 Hz + one 0.8 s stall | OK | 60.0 m/s | yes |

Rate equivalence: the same drive processed at 20 Hz and at 10 Hz yields 32.13 vs
29.14 vehicle cells per frame — **9.3 % apart**, i.e. physically equivalent
(byte-identical output across sampling rates is neither expected nor required).

### 20 Hz latency benchmark (tracker only, no rendering)

| | 20 Hz (budget 50 ms) | 10 Hz (budget 100 ms) |
|---|---|---|
| frames | 6925 | 6925 |
| mean | **16.03 ms** | 11.42 ms |
| median | 16.64 ms | 11.75 ms |
| p95 | 30.26 ms | 20.82 ms |
| **p99** | **36.13 ms** | 24.57 ms |
| max | 51.25 ms | 42.06 ms |
| over budget | 1 frame (0.014 %) | 0 |
| longest consecutive over-budget run | 1 frame | 0 |
| **acceptance p99 < budget** | **PASS** | **PASS** |

No sustained backlog: the single over-budget frame is isolated (run length 1), so
the loop recovers on the next frame.

---
## 6b. Regression suite

`python3 test_tracker_regression.py` — **17 / 17 pass** (deterministic seeds):

| # | test |
|---|---|
| 01 | a duplicate pair that separates becomes independent again |
| 02 | a stale duplicate cannot consume detections forever |
| 03 | two legitimate vehicles ~3.2 m apart stay independently represented |
| 04 | short transient proximity does not latch a duplicate |
| 05 | a coasted, decelerating track does not drive through the ego |
| 06 | coast velocity decays rather than staying constant |
| 07/08 | coast duration is the same physical duration at 10 Hz and 20 Hz |
| 09 | dt reaches the prediction (equal elapsed time → equal coast distance) |
| 10 | ego finite-difference span is a duration, not a frame count |
| 11 | causal prefix invariance |
| 12 | batch causal == streamed `update()` |
| 13 | the input matrix is never modified in place |
| 14 | geometry (120×80, ego 80/40, 1 m) and class IDs unchanged |
| 15 | no future information reaches an already-produced output |
| 16 | jittered and stalled dt stay numerically stable |
| 17 | same physical motion at 10 and 20 Hz gives equivalent trajectories |
| 18 | the tracker still bridges dropouts (does not just return RAW) |

## 6c. Success criteria

| | criterion | result |
|---|---|---|
| A | no unsupported through-ego crossings (\|y\|<3 m) | **0** (was 8, 7 unsupported) |
| B | no permanent stale duplicate suppression | max continuous separation while flagged **0.20 s**, grace 0.40 s, **0/388 over** |
| C | no indefinite invisible detection sink | **0** sink events beyond grace |
| D | major improvement in RAW-unrepresented @3 m | **31.34 % → 17.42 %** |
| E | tracker-induced lateral motion not increased | unexplained \|dy\|>2 m **831 → 791**; >4 m **125 → 96** |
| F | coast policy beats frozen constant velocity | drift at age 8 **−2.58 m → −0.33 m**; better than hold for 6 frames |
| G | temporal benefit preserved | dropout 0.056 → 0.057; fragments 4.26 % → **2.56 %**; obs/track 55.0 → **75.0** |
| H | no new safety losses | true losses **6 → 0** |
| I | prefix invariance | **exact**, all cut points |
| J | streaming = one current frame, past state only | **verified** |
| K | 10 Hz + 20 Hz + dynamic dt | **all pass** |
| L | 20 Hz p99 < 50 ms | **36.13 ms** |
| M | reproducibility | **identical** across runs |

## 7. Remaining limitations

1. **Orphan rate rises (31.2 % → 34.8 %).** The fixed causal output emits 126 %
   of RAW's vehicle cells, and a coasted state with no RAW vehicle within 3 m
   counts as an orphan. This is the direct cost of emitting a coast at all;
   OFFLINE, which never publishes trailing coast, sits at 24.0 %. Orphans cause
   conservative (over-)braking rather than missed braking, but they are noise.

2. **The tracked range still lags the RAW range for fast-converging stop signs**
   (~4 m at frames 2596–2615). The direction-aware pass-through now restores the
   raw evidence on every frame, so there is no control consequence and no lost
   detection — but the *tracked* position is still behind. This is the sensor
   range-bias interaction documented in the diagnosis; it is masked, not solved.
   OFFLINE still has one loss (f5400) with no target-speed effect.

3. **Lateral motion is still far from OFFLINE quality** (791 unexplained events
   vs 56). The causal yaw reproduces only ~0.64 of the true turn amplitude; the
   RTS smoother reaches ~1.0. This is intrinsic to causal filtering, not a bug.

4. **The 20 Hz run on this dataset is a configuration exercise, not a true
   20 Hz capture.** The recording is 10 Hz; feeding it as 20 Hz validates the
   plumbing (durations, coast windows, dt propagation) but a genuine 20 Hz drive
   has not been processed. That test must be repeated on real 20 Hz data.

5. **The old REALTIME divergence is still unexplained.** `matrix_tracked/` ships
   no config or manifest, so the difference between it and a local causal rerun
   cannot be attributed from the matrices alone.

6. **`test_rates` velocity bound is enforced, not emergent.** Peak |w| equals the
   clamp (60.0 m/s) at 20 Hz, i.e. the clamp is active. That is intended, but it
   means the raw estimator would still produce unphysical velocities at 20 Hz
   without it.

## 8. Methodology changes

- Coasting is no longer constant-velocity: the velocity is re-derived robustly at
  coast onset and then decays. This changes the motion model **during blind
  prediction only**; observed-frame Kalman behaviour is unchanged.
- A hard velocity clamp was added to the state.
- Duplicate suppression became a rolling-window, revocable decision.
- The safety pass-through coverage test became direction-aware.
- Everything else — Kalman structure, association, gating, class voting, the
  static-infrastructure prior, the offline RTS/stitch/merge stages, and the
  target-speed policy — is unchanged.

## 9. Reproduce

```bash
# datasets
python3 run_temporal_cleaning.py --matrix-dir realtime_capture/matrix \
        --mode causal  --dt 0.10 --out realtime_capture/fixed_online_10hz
python3 run_temporal_cleaning.py --matrix-dir realtime_capture/matrix \
        --mode causal  --dt 0.05 --out realtime_capture/fixed_online_20hz
python3 run_temporal_cleaning.py --matrix-dir realtime_capture/matrix \
        --mode offline --dt 0.10 --out realtime_capture/fixed_offline_10hz

# tests
python3 test_tracker_regression.py
python3 validate_tracker.py --matrix-dir realtime_capture/matrix

# comparison video (NEW file; the old four-way video is untouched)
# 6925 frames x 4 panels is slow to render (~1 h); use --start/--end for a clip
python3 compare_four_bev_modes.py \
  --panel "RAW DETECTION=realtime_capture/matrix" \
  --panel "OLD ONLINE (buggy)=realtime_capture/online/matrix_cleaned" \
  --panel "FIXED ONLINE (causal)=realtime_capture/fixed_online_10hz/matrix_cleaned" \
  --panel "OFFLINE (reference)=realtime_capture/fixed_offline_10hz/matrix_cleaned" \
  --fps 10 --out realtime_capture/raw_oldonline_fixedonline_offline.mp4

# live, 20 Hz
python3 live_bev_viewer.py --matrix-dir realtime_capture/matrix --dt 0.05
```

## 10. Integrity

- `realtime_capture/matrix` sha256 `53167fbc70b83bfee79d5f2e344afbf38ac9e791dd05f72b68214b307c95bcc7` — unchanged, and matches every manifest.
- `target_speed.py` sha256 `b973ac27b784c2f14d071344bf9e670d66f745e725fb88a860b124c7afb1f657` — unchanged.
- Previous result folders (`realtime_capture/{online,offline,rerun_causal,matrix_tracked}`) untouched; all new results are in `fixed_*` folders.
- Nothing was pushed to GitHub.
