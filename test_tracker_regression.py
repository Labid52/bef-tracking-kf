#!/usr/bin/env python3
"""
Regression suite for the BEV temporal tracker.

These tests lock in the behaviour of the fixes made after the root-cause
diagnosis, so the specific bugs cannot come back:

  * permanent/stale duplicate suppression and the "invisible detection sink"
  * blind constant-velocity coasting driving tracks through the ego
  * fixed 10 Hz timing assumptions

Run:
    python3 test_tracker_regression.py          # plain, no pytest needed
    python3 -m pytest test_tracker_regression.py -q
"""

import math
import numpy as np

import temporal_matrix_cleaner as tmc
import control_params as cp
from dataclasses import asdict

SEED = 12345
ROWS, COLS = tmc.ROWS, tmc.COLS


# ---------------------------------------------------------------- helpers ---
def blank():
    return np.zeros((ROWS, COLS), dtype=np.uint8)


def put(m, x_forward, y_right, cls=3):
    """Write one point detection at ego-relative metric coordinates."""
    r = tmc.EGO_ROW - int(round(x_forward))
    c = tmc.EGO_COL + int(round(y_right))
    if 0 <= r < ROWS and 0 <= c < COLS:
        m[r, c] = np.uint8(cls)
    return m


def scene(points, cls=3):
    m = blank()
    for x, y in points:
        put(m, x, y, cls)
    return m


def run_stream(frames, dt, cfg=None):
    c = tmc.OnlineTemporalCleaner(cfg)
    out = []
    for m in frames:
        cleaned, objs = c.update(m, dt=dt)
        out.append((cleaned, objs))
    return c, out


def veh_points(m):
    r, cc = np.nonzero(np.isin(m, [3, 4, 5, 6]))
    return sorted((float(tmc.EGO_ROW - a), float(b - tmc.EGO_COL))
                  for a, b in zip(r.tolist(), cc.tolist()))


# ------------------------------------------------------------------ tests ---
def test_01_duplicate_released_after_separation():
    """A pair that separates must stop being suppressed."""
    dt = 0.05
    frames = []
    for k in range(40):                      # 2.0 s co-located at 2 m apart
        frames.append(scene([(50.0, 0.0), (50.0, 2.0)]))
    for k in range(60):                      # then they separate steadily
        sep = 2.0 + 0.5 * k
        frames.append(scene([(50.0, 0.0), (50.0, min(sep, 35.0))]))
    c, out = run_stream(frames, dt)
    flagged = [t for t in c._main.active if t.duplicate_of is not None]
    assert not flagged, f"still suppressed after separating: {[t.tid for t in flagged]}"
    # both objects must be visible again at the end
    assert len(veh_points(out[-1][0])) >= 2, veh_points(out[-1][0])


def test_02_stale_duplicate_cannot_consume_forever():
    """A suppressed track must not stay an invisible detection sink."""
    dt = 0.05
    frames = [scene([(40.0, 0.0), (40.0, 2.0)]) for _ in range(40)]
    for k in range(80):
        frames.append(scene([(40.0, 0.0), (40.0, 2.0 + 0.4 * k)]))
    c, out = run_stream(frames, dt)
    n_emitted = len(veh_points(out[-1][0]))
    assert n_emitted >= 2, (
        f"only {n_emitted} object(s) emitted; a separated duplicate is still "
        "absorbing detections without being published")


def test_03_two_vehicles_3m_apart_stay_independent():
    """Genuine neighbours ~3 m apart must both survive (P1 root cause)."""
    dt = 0.05
    frames = []
    for k in range(200):
        x = 45.0 - 0.10 * k
        frames.append(scene([(x, -1.6), (x, 1.6)]))     # 3.2 m apart, parallel
    c, out = run_stream(frames, dt)
    pts = veh_points(out[-1][0])
    assert len(pts) >= 2, f"two parallel vehicles collapsed to {len(pts)}: {pts}"


def test_04_short_transient_proximity_does_not_suppress():
    """Crossing paths for a few frames must not latch a duplicate flag."""
    dt = 0.05
    frames = []
    for k in range(120):
        y = -8.0 + 0.14 * k                    # one sweeps across the other
        frames.append(scene([(40.0, 0.0), (40.0, y)]))
    c, out = run_stream(frames, dt)
    pts = veh_points(out[-1][0])
    assert len(pts) >= 2, f"transient proximity suppressed a track: {pts}"


def test_05_coast_does_not_drive_through_ego():
    """The diagnosed failure: object stops being detected, must not cross x=0."""
    dt = 0.05
    frames = []
    # A vehicle approaches and DECELERATES to a standstill relative to the ego,
    # exactly the geometry that produced the false crossings.
    xs = [30.0, 27.0, 24.5, 22.5, 21.0, 20.0, 19.4, 19.1, 19.0, 19.0, 19.0, 19.0]
    for x in xs:
        frames.append(scene([(x, 0.5)]))
    frames += [blank() for _ in range(40)]     # detection disappears entirely
    c, out = run_stream(frames, dt)
    for i, (cleaned, objs) in enumerate(out):
        for o in objs:
            assert o["x"] > 0.0, (
                f"frame {i}: emitted object crossed the ego at x={o['x']:.2f} "
                f"(provenance {o['provenance']})")


def test_06_coast_velocity_decays():
    """Coast displacement per frame must shrink, not stay constant."""
    dt = 0.05
    frames = [scene([(40.0 - 1.0 * k, 0.0)]) for k in range(10)]
    frames += [blank() for _ in range(20)]
    c, out = run_stream(frames, dt)
    xs = []
    for cleaned, objs in out[10:]:
        if objs:
            xs.append(objs[0]["x"])
    assert len(xs) >= 3, "no coasted frames emitted"
    steps = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    assert steps[-1] <= steps[0] + 1e-9, f"coast step did not decay: {steps}"


def test_07_08_coast_duration_is_physical_at_both_rates():
    """The same physical coast duration at 10 Hz and 20 Hz."""
    cfg = tmc.CleanerConfig()
    durations = {}
    for dt in (0.10, 0.05):
        frames = [scene([(40.0 - 0.5 * k, 0.0)]) for k in range(12)]
        frames += [blank() for _ in range(60)]
        c, out = run_stream(frames, dt)
        n_coast = 0
        for cleaned, objs in out[12:]:
            if any(o["provenance"] == tmc.PROV_COASTED for o in objs):
                n_coast += 1
            elif n_coast:
                break
        durations[dt] = n_coast * dt
    a, b = durations[0.10], durations[0.05]
    assert abs(a - b) <= 0.12, f"coast duration differs by rate: {durations}"
    assert a <= cfg.emit_coast_s + 0.12, f"coast longer than configured: {durations}"


def test_09_dynamic_dt_reaches_prediction():
    """dt must reach the prediction: equal ELAPSED time -> equal coast distance.

    A per-frame comparison would be wrong, because the coast decay is a time
    constant: after the same number of FRAMES the elapsed time differs.  The
    physical invariant is that over the same number of SECONDS the blind
    prediction travels the same distance at 10 Hz and at 20 Hz.
    """
    WINDOW_S = 0.20
    travel = {}
    for dt in (0.10, 0.05):
        n = int(round(1.2 / dt))
        frames = [scene([(40.0 - 8.0 * (k * dt), 0.0)]) for k in range(n)]
        frames += [blank() for _ in range(int(round(0.6 / dt)))]
        c, out = run_stream(frames, dt)
        xs = [o["x"] for cleaned, objs in out[n:] for o in objs]
        k = int(round(WINDOW_S / dt))
        assert len(xs) > k, f"not enough coast frames at dt={dt}: {len(xs)}"
        travel[dt] = abs(xs[k] - xs[0])
    assert travel[0.10] > 0.05 and travel[0.05] > 0.05, (
        f"coast did not move at all -> dt never reached the prediction: {travel}")
    rel = abs(travel[0.10] - travel[0.05]) / max(travel.values())
    assert rel < 0.35, (
        f"coast distance over {WINDOW_S}s differs by rate: {travel} ({100*rel:.0f}%)")


def test_10_ego_flow_baseline_uses_dt():
    """The finite-difference span is a physical duration, not a frame count."""
    cfg = tmc.CleanerConfig()
    t = tmc.Track(0, 0, tmc.Detection(x=40.0, y=0.0, cls=3, group="vehicle"), cfg)
    for k in range(1, 12):
        t.obs_frames.append(k)
        t.obs_xy.append((40.0 - 0.5 * k, 0.0))
    for dt, expect_gap in ((0.10, 2), (0.05, 4)):
        got = tmc._finite_difference_velocity(t, 11, cfg, dt)
        assert got is not None
        span_gap = int(round(cfg.ego_flow_baseline_s / dt))
        assert span_gap == expect_gap, (dt, span_gap, expect_gap)
        # velocity is -0.5 m per frame -> -0.5/dt m/s regardless of the gap used
        assert abs(got[2] - (-0.5 / dt)) < 1e-9, got


def test_11_causal_prefix_invariance():
    """Output over a prefix must not depend on later frames."""
    rs = np.random.RandomState(SEED)
    frames = []
    for k in range(220):
        pts = [(45.0 - 0.2 * k + rs.randn() * 0.4, -6.0 + rs.randn() * 0.3),
               (30.0 - 0.1 * k + rs.randn() * 0.4, 5.0 + rs.randn() * 0.3)]
        if k % 7 == 0:
            pts = pts[:1]                       # dropouts
        frames.append(scene(pts))
    _, full = run_stream(frames, 0.05)
    for cut in (17, 60, 111, 180):
        _, pre = run_stream(frames[:cut], 0.05)
        for i in range(cut):
            assert np.array_equal(pre[i][0], full[i][0]), f"prefix {cut} differs at {i}"


def test_12_batch_equals_stream():
    """clean_sequence(mode='causal') == repeated update() calls."""
    rs = np.random.RandomState(SEED + 1)
    frames = [scene([(40.0 - 0.15 * k + rs.randn() * 0.3, rs.randn() * 2.0)])
              for k in range(150)]
    arr = np.stack(frames)
    batch = tmc.clean_sequence(arr, mode="causal", verbose=False, dt=0.05).cleaned
    _, out = run_stream(frames, 0.05)
    stream = np.stack([o[0] for o in out])
    assert np.array_equal(batch, stream)


def test_13_input_matrix_not_modified():
    rs = np.random.RandomState(SEED + 2)
    frames = [scene([(40.0 - 0.2 * k, rs.randn())]) for k in range(30)]
    before = [m.copy() for m in frames]
    run_stream(frames, 0.05)
    for a, b in zip(before, frames):
        assert np.array_equal(a, b), "update() modified its input matrix"


def test_14_geometry_and_classes_unchanged():
    assert (tmc.ROWS, tmc.COLS) == (120, 80)
    assert tmc.EGO_ROW == 80 and tmc.EGO_COL == 40
    assert tmc.CELL_SIZE_M == 1.0
    m = blank(); put(m, 20.0, -3.0, 6)
    cleaned, _ = tmc.OnlineTemporalCleaner().update(m, dt=0.05)
    assert cleaned.shape == (120, 80) and cleaned.dtype == np.uint8
    assert set(np.unique(cleaned).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255}


def test_15_no_future_information():
    """Appending a future frame cannot change an already-produced output."""
    rs = np.random.RandomState(SEED + 3)
    frames = [scene([(40.0 - 0.2 * k, rs.randn() * 1.5)]) for k in range(80)]
    c1 = tmc.OnlineTemporalCleaner()
    got = [c1.update(m, dt=0.05)[0] for m in frames[:50]]
    c2 = tmc.OnlineTemporalCleaner()
    got2 = [c2.update(m, dt=0.05)[0] for m in frames]
    for i in range(50):
        assert np.array_equal(got[i], got2[i]), f"future frame changed output at {i}"


def test_16_variable_dt_is_stable():
    """Jittered and stalled dt must not explode the state."""
    rs = np.random.RandomState(SEED + 4)
    frames = [scene([(40.0 - 0.12 * k, rs.randn() * 1.2)]) for k in range(200)]
    c = tmc.OnlineTemporalCleaner()
    for k, m in enumerate(frames):
        dt = 0.05 + rs.randn() * 0.004
        if k == 120:
            dt = 3.0                        # a large scheduling stall
        cleaned, objs = c.update(m, dt=dt)
        for o in objs:
            assert np.isfinite(o["x"]) and np.isfinite(o["y"])
            assert abs(o["x"]) < 200 and abs(o["y"]) < 200, (k, o)
            assert abs(o["wx"]) < 200 and abs(o["wy"]) < 200, (k, o)
    for t in c._main.active:
        assert np.all(np.isfinite(t.cov)), "covariance became non-finite"


def test_17_rate_equivalence_physical():
    """Same physical motion sampled at 10 and 20 Hz -> equivalent trajectory."""
    res = {}
    for dt, n in ((0.10, 60), (0.05, 120)):
        frames = []
        for k in range(n):
            tt = k * dt
            frames.append(scene([(40.0 - 6.0 * tt, 3.0)]))   # 6 m/s closing
        c, out = run_stream(frames, dt)
        xs = [o["x"] for cleaned, objs in out for o in objs]
        res[dt] = xs[-1]
    assert abs(res[0.10] - res[0.05]) < 2.0, f"rate-dependent trajectory: {res}"


def test_18_still_better_than_raw():
    """The tracker must keep bridging dropouts (do not just return RAW)."""
    frames = []
    for k in range(80):
        if k % 4 == 3:
            frames.append(blank())                 # 25% dropout
        else:
            frames.append(scene([(40.0 - 0.25 * k, 2.0)]))
    c, out = run_stream(frames, 0.05)
    raw_hits = sum(1 for m in frames if np.isin(m, [3]).any())
    out_hits = sum(1 for cleaned, _ in out if np.isin(cleaned, [3]).any())
    assert out_hits > raw_hits, (
        f"tracker did not bridge any dropout: raw {raw_hits} vs out {out_hits}")


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            npass += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{npass}/{len(tests)} passed")
    return 0 if npass == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
