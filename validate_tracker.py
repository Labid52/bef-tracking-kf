#!/usr/bin/env python3
"""
Validation harness for the causal tracker: causality, rate handling and the
20 Hz real-time budget.  Complements test_tracker_regression.py, which covers
the behavioural regressions; this file covers the deployment properties that
need the real recorded sequence.

    python3 validate_tracker.py --matrix-dir realtime_capture/matrix
    python3 validate_tracker.py --matrix-dir realtime_capture/matrix --quick
"""

import argparse
import time
from pathlib import Path

import numpy as np

import control_params as cp
import temporal_matrix_cleaner as tmc


def load(matrix_dir: Path, limit=None):
    files = sorted(Path(matrix_dir).glob("*.npy"))
    if limit:
        files = files[:limit]
    return np.stack([np.load(p) for p in files])


def stream(frames, dt, cfg=None):
    c = tmc.OnlineTemporalCleaner(cfg)
    return np.stack([c.update(m, dt=dt)[0] for m in frames])


# --------------------------------------------------------------- causality --
def test_causality(frames, dt, cuts):
    print("\n-- CAUSALITY: prefix invariance ------------------------------------")
    full = stream(frames, dt)
    ok = True
    for k in cuts:
        if k >= len(frames):
            continue
        pre = stream(frames[:k], dt)
        same = np.array_equal(pre, full[:k])
        ok &= same
        print(f"   prefix 0..{k-1:<6} identical to the full run : {same}")
    print(f"   ALL PREFIXES IDENTICAL : {ok}")
    return ok


def test_batch_equals_stream(frames, dt):
    print("\n-- STREAMING: batch causal == repeated update() ---------------------")
    batch = tmc.clean_sequence(frames, mode="causal", verbose=False, dt=dt).cleaned
    st = stream(frames, dt)
    same = np.array_equal(batch, st)
    print(f"   clean_sequence(mode='causal') == OnlineTemporalCleaner loop : {same}")
    return same


def test_determinism(frames, dt):
    print("\n-- REPRODUCIBILITY -------------------------------------------------")
    a, b = stream(frames, dt), stream(frames, dt)
    same = np.array_equal(a, b)
    print(f"   two identical runs produce identical output : {same}")
    return same


# -------------------------------------------------------------------- rate --
def test_rates(frames):
    print("\n-- RATE HANDLING ---------------------------------------------------")
    rs = np.random.RandomState(7)
    cases = {
        "constant 10 Hz  (dt=0.100)": [0.10] * len(frames),
        "constant 20 Hz  (dt=0.050)": [0.05] * len(frames),
        "20 Hz + jitter  (+-4 ms)  ": list(0.05 + rs.randn(len(frames)) * 0.004),
        "20 Hz + one 0.8 s stall   ": [0.8 if i == len(frames)//2 else 0.05
                                       for i in range(len(frames))],
    }
    ok = True
    for label, dts in cases.items():
        c = tmc.OnlineTemporalCleaner()
        bad = 0
        maxspd = 0.0
        for m, dt in zip(frames, dts):
            _, objs = c.update(m, dt=dt)
            for o in objs:
                if not (np.isfinite(o["x"]) and np.isfinite(o["y"])
                        and abs(o["x"]) < 200 and abs(o["y"]) < 200):
                    bad += 1
                maxspd = max(maxspd, float(np.hypot(o["wx"], o["wy"])))
        cov_ok = all(np.all(np.isfinite(t.cov)) for t in c._main.active)
        good = (bad == 0) and cov_ok and maxspd < 100.0
        ok &= good
        print(f"   {label} : {'OK' if good else 'FAIL'}  "
              f"(bad states {bad}, max |w| {maxspd:5.1f} m/s, finite cov {cov_ok})")
    return ok


def test_rate_equivalence(frames):
    """Same physical span processed at 10 and 20 Hz must agree physically."""
    print("\n-- RATE EQUIVALENCE (physical, not byte-identical) -----------------")
    half = frames[::2]                       # 10 Hz view of the same drive
    a = stream(frames, 0.05)                 # treat full rate as 20 Hz
    b = stream(half, 0.10)                   # every other frame at 10 Hz
    na = np.isin(a, [3, 4, 5, 6]).sum() / len(a)
    nb = np.isin(b, [3, 4, 5, 6]).sum() / len(b)
    rel = abs(na - nb) / max(na, 1e-9)
    print(f"   vehicle cells per frame : 20 Hz {na:.2f}   10 Hz {nb:.2f}   "
          f"relative difference {100*rel:.1f}%")
    ok = rel < 0.25
    print(f"   physically equivalent (<25% difference) : {ok}")
    return ok


# ------------------------------------------------------------- performance --
def test_latency(frames, dt, budget_s):
    print(f"\n-- {1/dt:.0f} Hz LATENCY BENCHMARK (tracker only, no rendering) -------")
    c = tmc.OnlineTemporalCleaner()
    lat = np.empty(len(frames))
    for i, m in enumerate(frames):
        t0 = time.perf_counter()
        c.update(m, dt=dt)
        lat[i] = (time.perf_counter() - t0) * 1e3
    budget_ms = budget_s * 1e3
    over = lat > budget_ms
    run = best = 0
    for v in over:
        run = run + 1 if v else 0
        best = max(best, run)
    print(f"   frames            : {len(lat)}")
    print(f"   mean              : {lat.mean():7.2f} ms")
    print(f"   median            : {np.median(lat):7.2f} ms")
    print(f"   p95               : {np.percentile(lat,95):7.2f} ms")
    print(f"   p99               : {np.percentile(lat,99):7.2f} ms")
    print(f"   max               : {lat.max():7.2f} ms")
    print(f"   over {budget_ms:.0f} ms budget : {int(over.sum())} "
          f"({100*over.mean():.3f}%)")
    print(f"   longest consecutive over-budget run : {best} frame(s)")
    ok = np.percentile(lat, 99) < budget_ms
    print(f"   ACCEPTANCE p99 < {budget_ms:.0f} ms : {ok}")
    return ok, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-dir", type=Path, default=cp.MATRIX_DIR)
    ap.add_argument("--quick", action="store_true", help="use a 1500-frame subset")
    args = ap.parse_args()

    frames = load(args.matrix_dir, 1500 if args.quick else None)
    print(f"loaded {len(frames)} frames from {args.matrix_dir}")

    sub = frames[:1500]
    results = {}
    results["causality"] = test_causality(
        sub, 0.05, [37, 200, 651, 900, 1170, 1440])
    results["batch==stream"] = test_batch_equals_stream(sub, 0.05)
    results["determinism"] = test_determinism(sub, 0.05)
    results["rates"] = test_rates(sub)
    results["rate equivalence"] = test_rate_equivalence(sub)
    results["20 Hz latency"], _ = test_latency(frames, cp.LIVE_DT, cp.LIVE_DT)
    results["10 Hz latency"], _ = test_latency(frames, cp.LEGACY_DT, cp.LEGACY_DT)

    print("\n" + "=" * 68)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 68)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
