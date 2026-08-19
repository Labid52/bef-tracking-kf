#!/usr/bin/env python3
"""
Quantitative raw-vs-cleaned evaluation over the whole sequence.

Every metric is computed identically on both datasets by the SAME code, so the
comparison cannot be biased by measuring the two differently.  Metrics that
need object identity (track lifetime, fragmentation, reacquisition) use one
shared, deliberately simple reference associator applied to both inputs --
NOT the cleaner's own tracker, which would flatter the cleaned data.

    python3 compare_raw_cleaned.py
    python3 compare_raw_cleaned.py --cleaned-dir cleaned_causal --json report.json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import control_params
import temporal_matrix_cleaner as tmc
from target_speed import getTargetSpeed

HERE = Path(__file__).resolve().parent

# The target-speed policy is authoritative and is called here unmodified.
# Its arguments come from control_params so timing cannot drift between scripts.
TS_KW = dict(control_params.TARGET_SPEED_KW)

CLASS_NAME = {0: "empty", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
              5: "bus", 6: "truck", 7: "stop sign", 8: "traffic light",
              9: "red", 10: "yellow", 11: "green", 255: "unknown"}

# Reference associator: identical for raw and cleaned.
REF_GATE_M = 5.0
REF_GATE_PER_FRAME_M = 2.5
REF_MAX_GAP = 8


def load_dir(d: Path) -> np.ndarray:
    files = sorted(d.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy in {d}")
    return np.stack([np.load(p) for p in files])


def points(matrix):
    r, c = np.nonzero(matrix)
    return [(float(tmc.EGO_ROW - rr), float(cc - tmc.EGO_COL), int(matrix[rr, cc]))
            for rr, cc in zip(r.tolist(), c.tolist())]


# ---------------------------------------------------------------------------
# Reference association (shared by both datasets)
# ---------------------------------------------------------------------------

def reference_tracks(seq):
    """Constant-velocity greedy associator used only for measurement."""
    tracks, active = [], []
    for f in range(len(seq)):
        det = points(seq[f])
        cand = []
        for ti, t in enumerate(active):
            gap = f - t["last"]
            px = t["x"] + t["vx"] * gap
            py = t["y"] + t["vy"] * gap
            for di, (x, y, c) in enumerate(det):
                if tmc.class_group(c) != t["group"]:
                    continue
                d = math.hypot(px - x, py - y)
                if d <= REF_GATE_M + REF_GATE_PER_FRAME_M * gap:
                    cand.append((d, ti, di))
        cand.sort()
        ut, ud = set(), set()
        for d, ti, di in cand:
            if ti in ut or di in ud:
                continue
            ut.add(ti); ud.add(di)
            t = active[ti]; x, y, c = det[di]; gap = f - t["last"]
            t["vx"] = 0.6 * t["vx"] + 0.4 * (x - t["x"]) / gap
            t["vy"] = 0.6 * t["vy"] + 0.4 * (y - t["y"]) / gap
            t["x"], t["y"], t["last"] = x, y, f
            t["obs"].append(f); t["pos"].append((x, y)); t["cls"].append(c)
        for di, (x, y, c) in enumerate(det):
            if di in ud:
                continue
            active.append(dict(x=x, y=y, vx=0.0, vy=0.0, last=f, first=f,
                               group=tmc.class_group(c), obs=[f],
                               pos=[(x, y)], cls=[c]))
        keep = []
        for t in active:
            (tracks if f - t["last"] > REF_MAX_GAP else keep).append(t)
        active = keep
    return tracks + active


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def independent_yaw(seq, dt):
    """Per-frame ego yaw rate estimated from the RAW matrices alone.

    Used only by the evaluation, and computed once from the raw data, so the
    SAME physical allowance is applied to raw, offline and causal alike.  It
    reads the bearing rate directly (median over nearest-neighbour matched
    detections beyond 10 m), then averages over 3 frames because the source
    perception updates at 5 Hz and a 1-frame difference alternates between a
    held and a moved sample.
    """
    out = np.zeros(len(seq))
    for f in range(len(seq) - 1):
        a = [(p[0], p[1]) for p in points(seq[f])]
        b = [(p[0], p[1]) for p in points(seq[f + 1])]
        used, est = set(), []
        for x1, y1 in a:
            best, bd = None, 6.0
            for j, (x2, y2) in enumerate(b):
                if j in used:
                    continue
                d = math.hypot(x1 - x2, y1 - y2)
                if d < bd:
                    bd, best = d, j
            if best is None:
                continue
            used.add(best)
            x2, y2 = b[best]
            if abs(x1) < 10.0:
                continue
            est.append(-((y2 - y1) / dt) / x1)
        if len(est) >= 2:
            out[f] = float(np.median(est))
    return np.convolve(out, np.ones(3) / 3.0, "same")


def metrics(seq, label, psi_ind=None, dt=0.10):
    n_frames = len(seq)
    m = {"label": label, "frames": n_frames}

    # --- E: object-count stability -------------------------------------
    counts = np.array([len(points(seq[f])) for f in range(n_frames)])
    d = np.abs(np.diff(counts.astype(int)))
    m["count_mean"] = float(counts.mean())
    m["count_change_rate"] = float(np.mean(d > 0))
    m["count_change_mean_abs"] = float(d.mean())
    m["count_jump_ge2_rate"] = float(np.mean(d >= 2))

    tracks = reference_tracks(seq)
    m["ref_tracks"] = len(tracks)

    # --- A: disappearance / reappearance -------------------------------
    gaps = Counter()
    total_missing = total_obs = 0
    for t in tracks:
        o = t["obs"]
        total_obs += len(o)
        for i in range(1, len(o)):
            g = o[i] - o[i - 1] - 1
            if g > 0:
                gaps[g] += 1
                total_missing += g
    m["interior_gap_events"] = int(sum(gaps.values()))
    m["interior_gap_frames"] = int(total_missing)
    m["observations"] = int(total_obs)
    m["dropout_fraction"] = float(total_missing / max(total_missing + total_obs, 1))
    m["gap_hist"] = {int(k): int(v) for k, v in sorted(gaps.items())}
    # I: reacquisition after a short dropout, i.e. gaps of 1-3 frames that the
    # associator had to bridge.  Fewer is better: it means the data itself was
    # continuous.
    m["short_gap_events_1_3"] = int(sum(v for k, v in gaps.items() if 1 <= k <= 3))

    # --- B/C/D: frame-to-frame discontinuity within a track -------------
    dx, dy, dr, ux, uy = [], [], [], [], []
    for t in tracks:
        for i in range(1, len(t["obs"])):
            if t["obs"][i] - t["obs"][i - 1] != 1:
                continue
            ax, ay = t["pos"][i - 1]
            bx, by = t["pos"][i]
            dx.append(bx - ax); dy.append(by - ay)
            dr.append(math.hypot(bx - ax, by - ay))
            # How much displacement the EGO ROTATION alone could produce here:
            # |psi| * r * dt.  Subtracting it leaves motion the turn cannot
            # explain.  One-sided, so noise in psi can only make the test more
            # permissive -- it can never invent a jump.
            allow = 0.0
            if psi_ind is not None:
                f0 = t["obs"][i - 1]
                allow = abs(psi_ind[min(f0, len(psi_ind) - 1)]) * math.hypot(ax, ay) * dt
            ux.append(max(0.0, abs(bx - ax) - allow))
            uy.append(max(0.0, abs(by - ay) - allow))
    dx = np.abs(np.asarray(dx)); dy = np.abs(np.asarray(dy)); dr = np.asarray(dr)
    for name, arr in (("long", dx), ("lat", dy), ("total", dr)):
        if arr.size == 0:
            continue
        m[f"step_{name}_mean"] = float(arr.mean())
        m[f"step_{name}_p95"] = float(np.percentile(arr, 95))
        m[f"step_{name}_p99"] = float(np.percentile(arr, 99))
        m[f"step_{name}_max"] = float(arr.max())
    m["n_steps"] = int(dr.size)
    # J: physically implausible single-frame motion.  At 10 FPS, 3 m in one
    # frame is 30 m/s of relative motion; 2 m laterally is 20 m/s sideways.
    m["jump_long_gt3m"] = int((dx > 3.0).sum())
    m["jump_lat_gt2m"] = int((dy > 2.0).sum())
    m["jump_rate_long_gt3m"] = float((dx > 3.0).mean()) if dx.size else 0.0
    m["jump_rate_lat_gt2m"] = float((dy > 2.0).mean()) if dy.size else 0.0
    ux = np.asarray(ux); uy = np.asarray(uy)
    m["unexplained_long_gt3m"] = int((ux > 3.0).sum())
    m["unexplained_lat_gt2m"] = int((uy > 2.0).sum())
    # Second difference: a smooth trajectory has small curvature per frame.
    acc = []
    for t in tracks:
        p, o = t["pos"], t["obs"]
        for i in range(2, len(o)):
            if o[i] - o[i - 1] != 1 or o[i - 1] - o[i - 2] != 1:
                continue
            a = (p[i][0] - 2 * p[i - 1][0] + p[i - 2][0],
                 p[i][1] - 2 * p[i - 1][1] + p[i - 2][1])
            acc.append(math.hypot(*a))
    m["accel_mean_m_per_frame2"] = float(np.mean(acc)) if acc else 0.0
    m["accel_p95_m_per_frame2"] = float(np.percentile(acc, 95)) if acc else 0.0

    # --- G/H: track lifetime and fragmentation --------------------------
    lens = np.array([len(t["obs"]) for t in tracks])
    spans = np.array([t["obs"][-1] - t["obs"][0] + 1 for t in tracks])
    m["track_obs_mean"] = float(lens.mean()) if lens.size else 0.0
    m["track_obs_median"] = float(np.median(lens)) if lens.size else 0.0
    m["track_span_mean"] = float(spans.mean()) if spans.size else 0.0
    m["track_span_max"] = int(spans.max()) if spans.size else 0
    m["tracks_le2_obs"] = int((lens <= 2).sum())
    m["tracks_le2_obs_frac"] = float((lens <= 2).mean()) if lens.size else 0.0
    m["tracks_ge30_obs"] = int((lens >= 30).sum())
    # continuity = observations / span; 1.0 means never missing
    cont = lens / np.maximum(spans, 1)
    m["track_continuity_mean"] = float(cont.mean()) if lens.size else 0.0
    # class stability inside a track
    m["tracks_class_unstable"] = int(sum(1 for t in tracks if len(set(t["cls"])) > 1))
    m["tracks_class_unstable_frac"] = (
        float(sum(1 for t in tracks if len(set(t["cls"])) > 1) / len(tracks))
        if tracks else 0.0)

    # --- F: duplicate stop signs / red lights ---------------------------
    stop_mult = Counter()
    dup_pairs = 0
    dup_frames = 0
    seps = []
    for f in range(n_frames):
        pts = [p for p in points(seq[f]) if p[2] in (7, 9)]
        stop_mult[len(pts)] += 1
        hit = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                s = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                seps.append(s)
                if s <= 12.0:      # closer than any plausible pair of real signs
                    dup_pairs += 1
                    hit = True
        if hit:
            dup_frames += 1
    m["stop_count_hist"] = {int(k): int(v) for k, v in sorted(stop_mult.items())}
    m["frames_with_multiple_stops"] = int(sum(v for k, v in stop_mult.items() if k >= 2))
    m["stop_duplicate_pairs_le12m"] = int(dup_pairs)
    m["stop_duplicate_frames"] = int(dup_frames)
    m["stop_pair_sep_median"] = float(np.median(seps)) if seps else float("nan")

    far_pairs = 0
    for f in range(n_frames):
        pp = [p for p in points(seq[f]) if p[2] in (7, 9)]
        for i, a in enumerate(pp):
            for b in pp[i + 1:]:
                if math.hypot(a[0] - b[0], a[1] - b[1]) > 12.0:
                    far_pairs += 1
    m["frames_with_stop_pairs_gt12m"] = far_pairs

    # near-duplicate vehicles in one frame
    veh_dup = 0
    for f in range(n_frames):
        pts = [p for p in points(seq[f]) if tmc.class_group(p[2]) == tmc.GROUP_VEHICLE]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= 3.0:
                    veh_dup += 1
    m["vehicle_pairs_le3m"] = int(veh_dup)

    # A real duplicate is a SUSTAINED shadow, not two objects that pass close
    # to each other for a moment.  Measured on the shared reference tracks, so
    # gap-filling cannot inflate it.
    m.update(sustained_colocation(tracks))
    return m, tracks


def sustained_colocation(tracks, min_run=10):
    """Track pairs of the same group that stay mutually close for min_run frames."""
    out = {}
    for group, radius in ((tmc.GROUP_VEHICLE, 3.0), (tmc.GROUP_SIGN, 12.0)):
        by_frame = defaultdict(list)
        for k, t in enumerate(tracks):
            if t["group"] != group:
                continue
            for f, p in zip(t["obs"], t["pos"]):
                by_frame[f].append((k, p))
        runs = defaultdict(int)
        best = defaultdict(int)
        for f in sorted(by_frame):
            live = set()
            items = by_frame[f]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    (ka, pa), (kb, pb) = items[i], items[j]
                    if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= radius:
                        key = (min(ka, kb), max(ka, kb))
                        live.add(key)
                        runs[key] += 1
                        best[key] = max(best[key], runs[key])
            for key in list(runs):
                if key not in live:
                    runs[key] = 0
        out[f"sustained_pairs_{group}"] = int(sum(1 for v in best.values() if v >= min_run))
    return out


def target_speed_series(seq):
    return np.array([getTargetSpeed(matrix=seq[f], **TS_KW) for f in range(len(seq))])


def target_speed_metrics(ts, label):
    d = np.diff(ts)
    below = ts < 19.999
    episodes = []
    i = 0
    while i < len(ts):
        if below[i]:
            j = i
            while j < len(ts) and below[j]:
                j += 1
            episodes.append((i, j - i))
            i = j
        else:
            i += 1
    gaps = [episodes[k + 1][0] - (episodes[k][0] + episodes[k][1])
            for k in range(len(episodes) - 1)]
    return {
        "label": label,
        "frames_below_max": int(below.sum()),
        "mean_mph": float(ts.mean()),
        "abs_delta_mean_mph": float(np.abs(d).mean()),
        "abs_delta_p99_mph": float(np.percentile(np.abs(d), 99)),
        "abs_delta_max_mph": float(np.abs(d).max()),
        "jumps_gt5mph": int((np.abs(d) > 5).sum()),
        "jumps_gt10mph": int((np.abs(d) > 10).sum()),
        "braking_episodes": len(episodes),
        "episodes_1_frame": sum(1 for _, l in episodes if l == 1),
        "episodes_le3_frames": sum(1 for _, l in episodes if l <= 3),
        "episode_gaps_1_frame": sum(1 for g in gaps if g == 1),
        "episode_gaps_le3_frames": sum(1 for g in gaps if g <= 3),
    }


# ---------------------------------------------------------------------------
# Safety audit
# ---------------------------------------------------------------------------

def safety_audit(raw, cleaned, cfg):
    """Is every control-relevant raw detection still represented after cleaning?

    Reports the four categories separately:
      conservative : a counterpart of the same class group at the same distance
                     or nearer -- cleaning cannot have relaxed the target speed
      moved        : the only counterpart is >1.5 m FARTHER away
      restored     : the raw cell survives verbatim (safety pass-through)
      lost         : no counterpart at all -- a true audit failure
    """
    lost, conservative, moved, restored = [], 0, 0, 0
    tol = 6.0   # matches passthrough_cover_radius_m
    for f in range(len(raw)):
        cl = points(cleaned[f])
        exact = {(round(q[0]), round(q[1]), q[2]) for q in cl}
        for x, y, c in points(raw[f]):
            if not tmc._in_safety_zone(x, y, c, cfg):
                continue
            same = [q for q in cl if tmc.class_group(q[2]) == tmc.class_group(c)]
            near = [q for q in same if math.hypot(q[0] - x, q[1] - y) <= tol]
            # A counterpart that sits CLOSER to the ego and is still inside the
            # control region cannot relax the target speed, so it also counts as
            # safe representation even if it is further than the cover radius.
            closer = [q for q in same
                      if 0 < q[0] <= x and tmc._in_safety_zone(q[0], q[1], q[2], cfg)]
            if (round(x), round(y), c) in exact:
                restored += 1
                conservative += 1
            elif closer or (near and min(q[0] for q in near) <= x + 1.5):
                conservative += 1
            elif near:
                moved += 1
            else:
                lost.append((f, x, y, c))
    return conservative, moved, restored, lost


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, default=control_params.MATRIX_DIR)
    ap.add_argument("--cleaned-dir", type=Path, default=HERE / "cleaned" / "matrix_cleaned")
    ap.add_argument("--causal-dir", type=Path, default=HERE / "cleaned_causal" / "matrix_cleaned",
                    help="third column; pass 'none' to skip")
    ap.add_argument("--tracks", type=Path, default=HERE / "cleaned" / "tracks.npz",
                    help="sidecar with continuous sub-cell states")
    ap.add_argument("--egomotion", type=Path, default=HERE / "cleaned" / "egomotion.npz")
    ap.add_argument("--audit-failures", action="store_true",
                    help="enumerate and classify every remaining severe event")
    ap.add_argument("--resolution-study", action="store_true",
                    help="quantify how much residual jitter is 1 m quantisation")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--top", type=int, default=15,
                    help="how many worst target-speed differences to list")
    args = ap.parse_args()

    dt = control_params.DT
    cfg = tmc.CleanerConfig()

    datasets = [("raw", load_dir(args.matrix_dir)),
                ("offline", load_dir(args.cleaned_dir))]
    if str(args.causal_dir).lower() != "none" and args.causal_dir.exists():
        datasets.append(("causal", load_dir(args.causal_dir)))
    n = len(datasets[0][1])
    for name, d in datasets:
        if len(d) != n:
            raise ValueError(f"{name}: {len(d)} frames, expected {n}")

    raw = datasets[0][1]
    psi_ind = independent_yaw(raw, dt)

    M = {}
    TS = {}
    TSM = {}
    for name, d in datasets:
        M[name], _ = metrics(d, name, psi_ind=psi_ind, dt=dt)
        TS[name] = target_speed_series(d)
        TSM[name] = target_speed_metrics(TS[name], name)

    names = [n_ for n_, _ in datasets]
    W = 36
    CW = 12

    def hdr():
        print(f"  {'metric':<{W}}" + "".join(f"{x:>{CW}}" for x in names))

    def row(label, key, src=M, fmt="{}", better=None):
        vals = [src[n_][key] for n_ in names]
        cells = []
        for i, v in enumerate(vals):
            txt = fmt.format(v)
            if better and i > 0 and isinstance(v, (int, float)):
                base = vals[0]
                good = (v < base) if better == "lower" else (v > base)
                txt += "*" if good else " "
            cells.append(txt)
        print(f"  {label:<{W}}" + "".join(f"{c:>{CW}}" for c in cells))

    print("=" * (W + CW * len(names) + 4))
    print(f"RAW vs CLEANED  --  {n} frames, {n * dt:.1f} s @ {1/dt:.0f} FPS, dt = {dt:.2f} s")
    for name, _ in datasets:
        src = {"raw": args.matrix_dir, "offline": args.cleaned_dir,
               "causal": args.causal_dir}[name]
        print(f"  {name:<8}: {src}")
    print("  ('*' marks an improvement over raw; the same reference associator,")
    print("   independent of the cleaner, is applied to every column)")
    print("=" * (W + CW * len(names) + 4))
    hdr()

    print("\n-- E. object-count stability ---------------------------------------")
    row("objects per frame (mean)", "count_mean", fmt="{:.2f}")
    row("frames where count changes", "count_change_rate", fmt="{:.3f}", better="lower")
    row("mean |count change|", "count_change_mean_abs", fmt="{:.3f}", better="lower")
    row("P(|count change| >= 2)", "count_jump_ge2_rate", fmt="{:.3f}", better="lower")

    print("\n-- A / I. disappearance, reappearance, dropout ---------------------")
    row("reference tracks", "ref_tracks", better="lower")
    row("observations", "observations")
    row("interior gap events", "interior_gap_events", better="lower")
    row("interior missing frames", "interior_gap_frames", better="lower")
    row("dropout fraction of lifetime", "dropout_fraction", fmt="{:.4f}", better="lower")
    row("short (1-3 frame) dropouts", "short_gap_events_1_3", better="lower")

    print("\n-- B / C / D / J. frame-to-frame motion ----------------------------")
    row("consecutive-frame steps measured", "n_steps")
    row("longitudinal |dx| mean (m)", "step_long_mean", fmt="{:.3f}", better="lower")
    row("longitudinal |dx| p99 (m)", "step_long_p99", fmt="{:.2f}", better="lower")
    row("lateral |dy| mean (m)", "step_lat_mean", fmt="{:.3f}", better="lower")
    row("lateral |dy| p99 (m)", "step_lat_p99", fmt="{:.2f}", better="lower")
    row("long. jumps >3 m (naive)", "jump_long_gt3m", better="lower")
    row("lat. jumps >2 m (naive)", "jump_lat_gt2m", better="lower")
    print("  -- the two below subtract the displacement the ego TURN explains --")
    row("UNEXPLAINED long. jumps >3 m", "unexplained_long_gt3m", better="lower")
    row("UNEXPLAINED lat. jumps >2 m", "unexplained_lat_gt2m", better="lower")
    row("|2nd difference| mean (m)", "accel_mean_m_per_frame2", fmt="{:.3f}", better="lower")
    row("|2nd difference| p95 (m)", "accel_p95_m_per_frame2", fmt="{:.3f}", better="lower")

    print("\n-- G / H. track lifetime and fragmentation -------------------------")
    row("observations per track (mean)", "track_obs_mean", fmt="{:.2f}", better="higher")
    row("observations per track (median)", "track_obs_median", fmt="{:.1f}", better="higher")
    row("track span (mean frames)", "track_span_mean", fmt="{:.2f}", better="higher")
    row("tracks with >=30 observations", "tracks_ge30_obs", better="higher")
    row("fragment tracks (<=2 obs)", "tracks_le2_obs", better="lower")
    row("  as a fraction", "tracks_le2_obs_frac", fmt="{:.3f}", better="lower")
    row("track continuity (obs/span)", "track_continuity_mean", fmt="{:.3f}", better="higher")
    row("tracks with unstable class", "tracks_class_unstable", better="lower")

    print("\n-- F. duplicate stop signs / red lights ----------------------------")
    row("frames with >=2 stop/red", "frames_with_multiple_stops")
    row("duplicate pairs closer than 12 m", "stop_duplicate_pairs_le12m", better="lower")
    row("stop pairs >12 m apart (2 real signs)", "frames_with_stop_pairs_gt12m")
    row("vehicle pairs closer than 3 m", "vehicle_pairs_le3m")
    row("SUSTAINED duplicate vehicle pairs", "sustained_pairs_vehicle", better="lower")
    row("SUSTAINED duplicate sign pairs", "sustained_pairs_sign", better="lower")

    print("\n-- K. effect on getTargetSpeed (policy unmodified) ------------------")
    row("frames below max speed", "frames_below_max", src=TSM)
    row("mean target speed (mph)", "mean_mph", src=TSM, fmt="{:.3f}")
    row("mean |change| per frame (mph)", "abs_delta_mean_mph", src=TSM, fmt="{:.4f}", better="lower")
    row("p99 |change| (mph)", "abs_delta_p99_mph", src=TSM, fmt="{:.2f}", better="lower")
    row("changes > 5 mph", "jumps_gt5mph", src=TSM, better="lower")
    row("changes > 10 mph", "jumps_gt10mph", src=TSM, better="lower")
    row("braking episodes", "braking_episodes", src=TSM, better="lower")
    row("episodes lasting 1 frame", "episodes_1_frame", src=TSM, better="lower")
    row("episodes lasting <=3 frames", "episodes_le3_frames", src=TSM, better="lower")
    row("1-frame holes inside braking", "episode_gaps_1_frame", src=TSM, better="lower")

    # ---- per-dataset safety audit and target-speed differences ----
    for name, d in datasets[1:]:
        print(f"\n-- K2 / SAFETY for '{name}' ----------------------------------------")
        ts_r, ts_c = TS["raw"], TS[name]
        diff = np.nonzero(np.abs(ts_r - ts_c) > 1e-6)[0]
        higher = diff[ts_c[diff] > ts_r[diff]]
        lower = diff[ts_c[diff] < ts_r[diff]]
        print(f"  frames differing from raw    : {len(diff)} / {n}  ({100*len(diff)/n:.2f}%)")
        print(f"    brakes MORE than raw       : {len(lower)}")
        print(f"    brakes LESS than raw       : {len(higher)}")
        if len(higher):
            dh = ts_c[higher] - ts_r[higher]
            isolated = int(sum(1 for f in higher if 0 < f < n - 1
                               and ts_r[f - 1] > 19.999 and ts_r[f + 1] > 19.999))
            print(f"      of which raw was an isolated 1-frame spike: {isolated}")
            print(f"      relaxation median {np.median(dh):.2f} mph, p90 "
                  f"{np.percentile(dh, 90):.2f} mph, >5 mph in {int((dh > 5).sum())} frame(s)")
        rep, moved, restored, lost = safety_audit(raw, d, cfg)
        tot = rep + moved + len(lost)
        print(f"  control-relevant raw detections            : {tot}")
        print(f"    represented conservatively (same or nearer): {rep}")
        print(f"    counterpart moved >1.5 m farther away     : {moved}")
        print(f"    restored by the safety pass-through       : {restored}")
        print(f"    TRUE AUDIT FAILURES (no counterpart)      : {len(lost)}")
        for f, x, y, c in lost[:8]:
            solo = np.zeros((tmc.ROWS, tmc.COLS), dtype=np.uint8)
            solo[tmc.EGO_ROW - int(round(x)), tmc.EGO_COL + int(round(y))] = c
            print(f"      f{f:<6} {CLASS_NAME.get(c, c):<10} ({x:+.0f},{y:+.0f}) m -> "
                  f"would alone command {getTargetSpeed(matrix=solo, **TS_KW):.2f} mph")

    print(f"\n  worst {args.top} raw-vs-offline target-speed differences:")
    ts_r, ts_c = TS["raw"], TS["offline"]
    diff = np.nonzero(np.abs(ts_r - ts_c) > 1e-6)[0]
    print(f"    {'frame':>6} {'raw':>7} {'offline':>8}  controlling object (raw -> offline)")
    for f in sorted(diff, key=lambda f: -abs(ts_r[f] - ts_c[f]))[:args.top]:
        print(f"    {f:>6} {ts_r[f]:>7.2f} {ts_c[f]:>8.2f}  "
              f"{_controller(raw[f]):<26} -> {_controller(datasets[1][1][f])}")

    if args.audit_failures:
        audit_failures(raw, datasets[1][1], args.tracks, args.egomotion, cfg, dt,
                       top=args.top)

    if args.resolution_study:
        resolution_study(datasets[1][1], args.tracks)

    if args.json:
        out = {"frames": n, "dt": dt,
               "metrics": {k: v for k, v in M.items()},
               "target_speed": {k: v for k, v in TSM.items()}}
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\n  wrote {args.json}")


# ---------------------------------------------------------------------------
# Residual failure audit (Phase 2)
# ---------------------------------------------------------------------------

#: Cause taxonomy.  Every severe event is assigned exactly one of these, by the
#: first matching rule in the documented priority order in classify_jump().
CAUSES = (
    "legitimate physical motion", "raw range/depth error", "quantisation",
    "association error", "track fragmentation", "duplicate hypothesis",
    "class error", "safety pass-through", "corridor-boundary crossing",
    "hood-filter interaction", "ego-yaw estimation error", "unavoidable ambiguity",
)


def load_sidecar(path: Path):
    """Sidecar -> {frame: [object dicts]} with continuous state and provenance."""
    tk = np.load(path, allow_pickle=True)
    prov_names = [str(x) for x in tk["provenance_names"]]
    by_frame = defaultdict(list)
    for i in range(len(tk["frame"])):
        by_frame[int(tk["frame"][i])].append(dict(
            tid=int(tk["track_id"][i]), cls=int(tk["cls"][i]),
            x=float(tk["x_forward_m"][i]), y=float(tk["y_right_m"][i]),
            prov=prov_names[int(tk["provenance_id"][i])]))
    return by_frame


def _nearest(items, x, y, tol):
    best, bd = None, tol
    for it in items:
        d = math.hypot(it["x"] - x, it["y"] - y)
        if d <= bd:
            best, bd = it, d
    return best


def _nearest_raw(seq, f, x, y, group, tol):
    best, bd = None, tol
    for px, py, pc in points(seq[f]):
        if tmc.class_group(pc) != group:
            continue
        d = math.hypot(px - x, py - y)
        if d <= bd:
            best, bd = (px, py, pc), d
    return best


def classify_jump(raw, side, ego_psi, f, pa, pb, dt):
    """Classify one large frame-to-frame displacement.

    Priority order (first match wins), chosen so an explanation that is
    *demonstrable from the data* always beats a residual catch-all:
      1 a safety pass-through endpoint (raw evidence written verbatim)
      2 the reference associator crossed internal track identities
      3 the continuous state barely moved -- the cell changed, not the object
      4 the raw matrix shows the same jump -> the cleaner is faithfully
        following the measurement, i.e. sensor range/projection error
      5 the motion matches the estimated ego yaw rate -> legitimate
      6 the class label changed across the step
      7 an endpoint is a prediction rather than an observation
    """
    ax, ay = pa["x"], pa["y"]
    bx, by = pb["x"], pb["y"]
    jump = math.hypot(bx - ax, by - ay)
    er, et = tmc._radial_tangential_error((ax, ay), (bx, by))
    ev = dict(frame=f, jump=jump, er=er, et=et,
              prov=(pa["prov"], pb["prov"]), tid=(pa["tid"], pb["tid"]),
              cls=(pa["cls"], pb["cls"]))

    if tmc.PROV_PASSTHROUGH in (pa["prov"], pb["prov"]):
        return "safety pass-through", ev
    if pa["tid"] != pb["tid"]:
        seen_a = any(o["tid"] == pa["tid"] for o in side.get(f + 1, []))
        seen_b = any(o["tid"] == pb["tid"] for o in side.get(f, []))
        return ("association error" if (seen_a and seen_b)
                else "track fragmentation"), ev

    cont = jump
    rast = math.hypot(round(bx) - round(ax), round(by) - round(ay))
    ev["continuous_jump"] = cont
    if rast >= 1.0 and cont < 0.6:
        return "quantisation", ev

    group = tmc.class_group(pa["cls"])
    ra = _nearest_raw(raw, f, ax, ay, group, 6.0)
    rb = _nearest_raw(raw, f + 1, bx, by, group, 6.0)
    if ra and rb:
        raw_jump = math.hypot(rb[0] - ra[0], rb[1] - ra[1])
        ev["raw_jump"] = raw_jump
        if raw_jump >= 0.7 * jump:
            return "raw range/depth error", ev

    psi = float(ego_psi[min(f, len(ego_psi) - 1)])
    yaw_lat = abs(psi) * math.hypot(ax, ay) * dt
    ev["yaw_expected"] = yaw_lat
    if et > 2.0 * er and yaw_lat > 0.5 * et:
        return "legitimate physical motion", ev
    if et > 2.0 * er and abs(psi) > 0.15:
        return "ego-yaw estimation error", ev
    if pa["cls"] != pb["cls"]:
        return "class error", ev
    if tmc.PROV_OBSERVED not in (pa["prov"], pb["prov"]):
        return "unavoidable ambiguity", ev
    if er > 2.0 * et:
        return "raw range/depth error", ev
    return "unavoidable ambiguity", ev


def _controller_full(matrix):
    """(margin, cls, x, y) of the detection currently setting the target speed."""
    best = None
    for x, y, c in points(matrix):
        if x <= 0:
            continue
        if c in tmc.TS_OBSTACLE_CLASSES and abs(y) <= TS_KW["width_of_interest_m"] / 2:
            if c == 3 and x <= TS_KW["hood_ignore_distance_m"]:
                continue
            key = (x - TS_KW["safety_stopping_distance_m"], c, x, y)
        elif c in tmc.TS_STOP_CLASSES and abs(y) <= TS_KW["stop_lateral_limit_m"]:
            key = (x - TS_KW["stop_offset_m"], c, x, y)
        else:
            continue
        if best is None or key[0] < best[0]:
            best = key
    return best


def _fmt_ctrl(c):
    return "none" if c is None else f"{CLASS_NAME.get(c[1], c[1])}({c[2]:+.0f},{c[3]:+.0f})"


def _classify_ts_event(side, f, ca, cb):
    """Why did the controlling object change between f and f+1?"""
    for c, fr in ((ca, f), (cb, f + 1)):
        if c is None:
            continue
        o = _nearest(side.get(fr, []), c[2], c[3], 1.5)
        if o and o["prov"] == tmc.PROV_PASSTHROUGH:
            return "safety pass-through"
    live = ca or cb
    if live is None:
        return "unavoidable ambiguity"
    gone = cb if ca else ca
    x, y = live[2], live[3]
    if gone is None:
        near = _nearest(side.get(f + 1 if ca else f, []), x, y, 6.0)
        if near is not None:
            lim = (TS_KW["stop_lateral_limit_m"] if live[1] in tmc.TS_STOP_CLASSES
                   else TS_KW["width_of_interest_m"] / 2)
            if abs(near["y"]) > lim or near["x"] <= 0:
                return "corridor-boundary crossing"
            if live[1] == 3 and near["x"] <= TS_KW["hood_ignore_distance_m"]:
                return "hood-filter interaction"
            return "track fragmentation"
        if x <= 1.5:
            return "corridor-boundary crossing"
        return "track fragmentation"
    return "legitimate physical motion"


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, tuple):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    return str(v)


def audit_failures(raw, cleaned, tracks_path, ego_path, cfg, dt, top=0):
    """Enumerate EVERY remaining severe event in ``cleaned`` and classify it."""
    n = len(raw)
    side = load_sidecar(tracks_path)
    ego_psi = (np.load(ego_path)["ego_yaw_rate_radps"] if ego_path.exists()
               else np.zeros(n))
    tracks = reference_tracks(cleaned)

    print("\n" + "=" * 78)
    print("RESIDUAL FAILURE AUDIT -- every remaining severe event, classified")
    print("=" * 78)
    buckets = defaultdict(list)

    for t in tracks:
        for i in range(1, len(t["obs"])):
            if t["obs"][i] - t["obs"][i - 1] != 1:
                continue
            f = t["obs"][i - 1]
            ax, ay = t["pos"][i - 1]
            bx, by = t["pos"][i]
            big_long = abs(bx - ax) > 3.0
            big_lat = abs(by - ay) > 2.0
            if not (big_long or big_lat):
                continue
            kind = ("longitudinal >3 m + lateral >2 m" if (big_long and big_lat)
                    else "longitudinal >3 m" if big_long else "lateral >2 m")
            pa = _nearest(side.get(f, []), ax, ay, 1.5)
            pb = _nearest(side.get(f + 1, []), bx, by, 1.5)
            if pa is None or pb is None:
                buckets[kind].append(("unavoidable ambiguity",
                                      dict(frame=f, note="no sidecar match")))
                continue
            cause, ev = classify_jump(raw, side, ego_psi, f, pa, pb, dt)
            ev["dx"], ev["dy"] = bx - ax, by - ay
            buckets[kind].append((cause, ev))

    for f in range(n):
        pts = [p for p in points(cleaned[f]) if p[2] in (7, 9)]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d > 12.0:
                    continue
                oa = _nearest(side.get(f, []), pts[i][0], pts[i][1], 1.5)
                ob = _nearest(side.get(f, []), pts[j][0], pts[j][1], 1.5)
                provs = tuple(o["prov"] if o else "?" for o in (oa, ob))
                tids = tuple(o["tid"] if o else None for o in (oa, ob))
                cause = ("safety pass-through" if tmc.PROV_PASSTHROUGH in provs
                         else "unavoidable ambiguity" if None in tids
                         else "duplicate hypothesis")
                buckets["duplicate stop/red pair <=12 m"].append(
                    (cause, dict(frame=f, sep=d, prov=provs, tid=tids)))

    ts = target_speed_series(cleaned)
    for f in range(n - 1):
        if abs(ts[f + 1] - ts[f]) <= 10.0:
            continue
        ca, cb = _controller_full(cleaned[f]), _controller_full(cleaned[f + 1])
        buckets["target-speed jump >10 mph"].append(
            (_classify_ts_event(side, f, ca, cb),
             dict(frame=f, ts=(float(ts[f]), float(ts[f + 1])),
                  before=_fmt_ctrl(ca), after=_fmt_ctrl(cb))))

    below = ts < TS_KW["max_target_speed_mph"] - 1e-3
    for f in range(1, n - 1):
        if below[f] and not below[f - 1] and not below[f + 1]:
            c = _controller_full(cleaned[f])
            o = _nearest(side.get(f, []), c[2], c[3], 1.5) if c else None
            cause = ("safety pass-through"
                     if o and o["prov"] == tmc.PROV_PASSTHROUGH
                     else _classify_ts_event(side, f, None, c))
            buckets["isolated 1-frame braking"].append(
                (cause, dict(frame=f, ts=float(ts[f]), object=_fmt_ctrl(c))))

    total = 0
    for kind in sorted(buckets):
        evs = buckets[kind]
        total += len(evs)
        print(f"\n{kind}: {len(evs)} event(s)")
        for cause, k in Counter(c for c, _ in evs).most_common():
            print(f"    {k:4d}  {cause}")
        for cause, ev in (evs[:top] if top else []):
            det = ", ".join(f"{a}={_fmt(b)}" for a, b in ev.items() if a != "frame")
            print(f"      f{ev['frame']:<6} {cause:<28} {det}")

    print(f"\nTOTAL SEVERE EVENTS AUDITED: {total}")
    grand = Counter()
    for evs in buckets.values():
        grand.update(c for c, _ in evs)
    print("cause distribution across all event types:")
    for cause, k in grand.most_common():
        print(f"  {k:5d}  ({100 * k / max(total, 1):5.1f}%)  {cause}")
    return buckets


def resolution_study(cleaned, tracks_path: Path):
    """Answer the cell-size question with evidence.

    Compares each track's CONTINUOUS state against the same state rounded onto
    the original 1 m grid, and the target speed the policy produces from each.
    The policy formulas are identical; only the position resolution differs.
    """
    tk = np.load(tracks_path, allow_pickle=True)
    keep = tk["track_id"] >= 0
    f, tid = tk["frame"][keep], tk["track_id"][keep]
    cls = tk["cls"][keep]
    x, y = tk["x_forward_m"][keep], tk["y_right_m"][keep]
    o = np.lexsort((f, tid))
    f, tid, cls, x, y = f[o], tid[o], cls[o], x[o], y[o]
    rx, ry = np.round(x), np.round(y)
    step = (tid[1:] == tid[:-1]) & (f[1:] == f[:-1] + 1)

    print("\n-- CELL-RESOLUTION STUDY --------------------------------------------")
    print(f"  {'quantity':<36} {'continuous':>12} {'1 m raster':>12}")
    for nm, a, b in (("|dx| per frame, mean (m)", np.abs(np.diff(x))[step], np.abs(np.diff(rx))[step]),
                     ("|dy| per frame, mean (m)", np.abs(np.diff(y))[step], np.abs(np.diff(ry))[step])):
        print(f"  {nm:<36} {a.mean():>12.3f} {b.mean():>12.3f}")
    s2 = (tid[2:] == tid[:-2]) & (f[2:] == f[:-2] + 2)
    for nm, pair in (("|2nd difference| x, mean (m)", (x, rx)),
                     ("|2nd difference| y, mean (m)", (y, ry))):
        c = np.abs(pair[0][2:] - 2 * pair[0][1:-1] + pair[0][:-2])[s2]
        q = np.abs(pair[1][2:] - 2 * pair[1][1:-1] + pair[1][:-2])[s2]
        print(f"  {nm:<36} {c.mean():>12.4f} {q.mean():>12.4f}")
    n_step = int(step.sum())
    cdx, qdx = np.abs(np.diff(x))[step], np.abs(np.diff(rx))[step]
    print(f"  steps where the cell changes but the object moved <0.5 m : "
          f"{int(((qdx > 0) & (cdx < 0.5)).sum())} / {n_step}")
    print(f"  steps where the object moved >0.5 m but the cell did not : "
          f"{int(((qdx == 0) & (cdx > 0.5)).sum())} / {n_step}")

    mph = 1.0 / 0.44704

    def ts_continuous(fr):
        m = f == fr
        best = TS_KW["max_target_speed_mph"]
        for c, xx, yy in zip(cls[m], x[m], y[m]):
            if xx <= 0:
                continue
            if c in tmc.TS_OBSTACLE_CLASSES and abs(yy) <= TS_KW["width_of_interest_m"] / 2:
                if c == 3 and xx <= TS_KW["hood_ignore_distance_m"]:
                    continue
                v = math.sqrt(2 * TS_KW["deceleration_mps2"]
                              * max(0.0, xx - TS_KW["safety_stopping_distance_m"])) * mph
            elif c in tmc.TS_STOP_CLASSES and abs(yy) <= TS_KW["stop_lateral_limit_m"]:
                v = math.sqrt(2 * TS_KW["deceleration_mps2"]
                              * max(0.0, xx - TS_KW["stop_offset_m"])) * mph
            else:
                continue
            best = min(best, v)
        return best

    ts_grid = target_speed_series(cleaned)
    ts_cont = np.array([ts_continuous(i) for i in range(len(cleaned))])
    d = np.abs(ts_grid - ts_cont)
    print(f"\n  target speed, 1 m raster vs continuous state (same policy formulas):")
    print(f"    frames differing at all       : {int((d > 1e-6).sum())} / {len(cleaned)}")
    print(f"    |difference| mean / p95 / max : {d.mean():.4f} / "
          f"{np.percentile(d, 95):.3f} / {d.max():.3f} mph")
    print(f"    frames differing by > 1 mph   : {int((d > 1.0).sum())}")
    print(f"    mean |change| per frame       : raster {np.abs(np.diff(ts_grid)).mean():.4f}"
          f"   continuous {np.abs(np.diff(ts_cont)).mean():.4f} mph")


def _controller(matrix):
    """Which detection is currently setting the target speed, if any."""
    best = None
    for x, y, c in points(matrix):
        if x <= 0:
            continue
        if c in tmc.TS_OBSTACLE_CLASSES and abs(y) <= 2.0:
            if c == 3 and x <= 6.0:
                continue
            key = (x - 8.0, c, x, y)
        elif c in tmc.TS_STOP_CLASSES and abs(y) <= 15.0:
            key = (x - 5.0, c, x, y)
        else:
            continue
        if best is None or key[0] < best[0]:
            best = key
    if best is None:
        return "none (max speed)"
    _, c, x, y = best
    return f"{CLASS_NAME.get(c, c)} at ({x:+.0f}, {y:+.0f}) m"


if __name__ == "__main__":
    main()
