#!/usr/bin/env python3
"""
BEV validation animation: raw detections vs cleaned tracks, with the real
target speed from the unmodified getTargetSpeed().

Every input frame appears exactly once at 10 FPS, so an N-frame sequence
produces an N/10 second video.

    python3 animate_target_speed_bev_cleaned.py
    python3 animate_target_speed_bev_cleaned.py --start 1140 --end 1200 \
            --out cleaned/worst_case_1140.mp4

Left panel  : raw matrix, one marker per non-empty cell.
Right panel : cleaned tracks at their continuous sub-cell positions, each with
              a persistent id, a short motion trail and a velocity arrow.
Bottom      : target speed computed by getTargetSpeed() on the raw matrix and
              on the cleaned matrix, plus which object is controlling.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

import control_params
import temporal_matrix_cleaner as tmc
from target_speed import getTargetSpeed

HERE = Path(__file__).resolve().parent

TS_KW = dict(control_params.TARGET_SPEED_KW)

CLASS_STYLE = {
    1: ("person",       "#e8453c", "o"),
    2: ("bicycle",      "#e8853c", "o"),
    3: ("car",          "#3c8ce8", "s"),
    4: ("motorcycle",   "#8c3ce8", "s"),
    5: ("bus",          "#00a0a0", "s"),
    6: ("truck",        "#1f5fa0", "s"),
    7: ("stop sign",    "#d40000", "H"),
    8: ("traffic light", "#888888", "^"),
    9: ("red light",    "#d40000", "^"),
    10: ("yellow light", "#d0a000", "^"),
    11: ("green light", "#22a022", "^"),
}
DEFAULT_STYLE = ("unknown", "#999999", "x")

#: Diagnostic ring colour per provenance (only drawn with --show-provenance).
PROV_RING = {
    tmc.PROV_INTERPOLATED: "#1f77b4",
    tmc.PROV_COASTED: "#ff7f0e",
    tmc.PROV_PASSTHROUGH: "#2ca02c",
}
TRAIL = 12


def style(cls):
    return CLASS_STYLE.get(int(cls), DEFAULT_STYLE)


def controlling_object(matrix):
    """Which detection sets the target speed for this matrix, if any."""
    best = None
    r, c = np.nonzero(matrix)
    for rr, cc in zip(r.tolist(), c.tolist()):
        cls = int(matrix[rr, cc])
        x = float(tmc.EGO_ROW - rr)
        y = float(cc - tmc.EGO_COL)
        if x <= 0:
            continue
        if cls in tmc.TS_OBSTACLE_CLASSES and abs(y) <= TS_KW["width_of_interest_m"] / 2:
            if cls == 3 and x <= TS_KW["hood_ignore_distance_m"]:
                continue
            margin = x - TS_KW["safety_stopping_distance_m"]
        elif cls in tmc.TS_STOP_CLASSES and abs(y) <= TS_KW["stop_lateral_limit_m"]:
            margin = x - TS_KW["stop_offset_m"]
        else:
            continue
        if best is None or margin < best[0]:
            best = (margin, cls, x, y)
    return best


def setup_axis(ax, title):
    ax.set_xlim(tmc.Y_MIN - 1, tmc.Y_MAX + 1)
    ax.set_ylim(tmc.X_MIN - 1, tmc.X_MAX + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral  y  [m]   (right positive)")
    ax.set_ylabel("forward  x  [m]")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    # target-speed regions, drawn from the SAME parameters the policy uses
    hw = TS_KW["width_of_interest_m"] / 2
    ax.add_patch(Rectangle((-hw, 0), 2 * hw, tmc.X_MAX, color="#3c8ce8",
                           alpha=0.07, zorder=0))
    sl = TS_KW["stop_lateral_limit_m"]
    ax.add_patch(Rectangle((-sl, 0), 2 * sl, tmc.X_MAX, color="#d40000",
                           alpha=0.035, zorder=0))
    ax.plot(0, 0, marker="^", ms=13, color="black", zorder=6)
    ax.text(0, -6, "ego", ha="center", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, default=control_params.MATRIX_DIR)
    ap.add_argument("--cleaned-dir", type=Path, default=HERE / "cleaned" / "matrix_cleaned")
    ap.add_argument("--tracks", type=Path, default=HERE / "cleaned" / "tracks.npz")
    ap.add_argument("--out", type=Path, default=HERE / "cleaned" / "bev_raw_vs_cleaned.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=int, default=int(control_params.FPS),
                    help="must equal the data rate (10 FPS, frame step 1)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--show-provenance", action="store_true",
                    help="diagnostic: ring non-observed states (interpolated / "
                         "coasted / safety pass-through). Off by default so the "
                         "presentation animation stays visually continuous.")
    args = ap.parse_args()

    raw_files = sorted(args.matrix_dir.glob("*.npy"))
    cln_files = sorted(args.cleaned_dir.glob("*.npy"))
    if len(raw_files) != len(cln_files):
        raise ValueError("raw and cleaned frame counts differ")
    lo = args.start
    hi = args.end if args.end is not None else len(raw_files)
    idx = list(range(lo, hi))
    if not idx:
        raise ValueError("empty frame range")

    raw = np.stack([np.load(raw_files[f]) for f in idx])
    cln = np.stack([np.load(cln_files[f]) for f in idx])

    tk = np.load(args.tracks, allow_pickle=True)
    prov_names = ([str(v) for v in tk["provenance_names"]]
                  if "provenance_names" in tk else list(tmc.PROVENANCES))
    prov_id = tk["provenance_id"] if "provenance_id" in tk else None
    by_frame = {}
    for i, (f, tid, cls, x, y, vx, vy, ob, st) in enumerate(zip(
            tk["frame"], tk["track_id"], tk["cls"], tk["x_forward_m"],
            tk["y_right_m"], tk["vx_mps"], tk["vy_mps"], tk["observed"],
            tk["is_static"])):
        prov = (prov_names[int(prov_id[i])] if prov_id is not None
                else (tmc.PROV_OBSERVED if ob else tmc.PROV_COASTED))
        by_frame.setdefault(int(f), []).append(
            (int(tid), int(cls), float(x), float(y), float(vx), float(vy),
             bool(ob), bool(st), prov))
    history = {}
    for f in sorted(by_frame):
        for tid, cls, x, y, *_ in by_frame[f]:
            history.setdefault(tid, []).append((f, x, y))

    ts_raw = np.array([getTargetSpeed(matrix=m, **TS_KW) for m in raw])
    ts_cln = np.array([getTargetSpeed(matrix=m, **TS_KW) for m in cln])

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.34, wspace=0.18,
                          left=0.06, right=0.985, top=0.935, bottom=0.135)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cln = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])
    setup_axis(ax_raw, "RAW semantic matrix  (one marker per non-empty cell)")
    setup_axis(ax_cln, "CLEANED tracks  (one marker per tracked object)")

    ax_ts.set_xlim(idx[0], idx[-1])
    ax_ts.set_ylim(-1, 21.5)
    ax_ts.set_xlabel("frame")
    ax_ts.set_ylabel("target speed [mph]")
    ax_ts.grid(alpha=0.25, linewidth=0.5)
    ax_ts.plot(idx, ts_raw, color="#b0b0b0", lw=1.0, label="getTargetSpeed(raw)")
    ax_ts.plot(idx, ts_cln, color="#c62828", lw=1.6, label="getTargetSpeed(cleaned)")
    ax_ts.legend(loc="lower left", fontsize=8, ncol=2)
    cursor = ax_ts.axvline(idx[0], color="black", lw=1.0)

    hdr = fig.text(0.5, 0.99, "", ha="center", va="top", fontsize=11,
                   family="monospace")
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9)
    info_raw = ax_raw.text(0.02, 0.985, "", transform=ax_raw.transAxes,
                           va="top", fontsize=8.5, family="monospace", bbox=box)
    info_cln = ax_cln.text(0.02, 0.985, "", transform=ax_cln.transAxes,
                           va="top", fontsize=8.5, family="monospace", bbox=box)

    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=6, label=nm)
               for nm, c, mk in CLASS_STYLE.values()]
    if args.show_provenance:
        for nm2, col2 in (("interpolated", PROV_RING[tmc.PROV_INTERPOLATED]),
                          ("coasted", PROV_RING[tmc.PROV_COASTED]),
                          ("safety pass-through", PROV_RING[tmc.PROV_PASSTHROUGH])):
            handles.append(plt.Line2D([], [], color=col2, marker="o", ls="", ms=8,
                                      mfc="none", mec=col2, label=nm2))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               fontsize=7, framealpha=0.9, ncol=6)

    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def draw(i):
        clear()
        f = idx[i]
        cursor.set_xdata([f, f])

        # ---- raw ----
        r, c = np.nonzero(raw[i])
        for rr, cc in zip(r.tolist(), c.tolist()):
            cls = int(raw[i][rr, cc])
            nm, col, mk = style(cls)
            artists.append(ax_raw.plot(cc - tmc.EGO_COL, tmc.EGO_ROW - rr,
                                       marker=mk, color=col, ms=8, ls="",
                                       mec="black", mew=0.4, zorder=4)[0])
        cr = controlling_object(raw[i])
        if cr:
            artists.append(ax_raw.plot(cr[3], cr[2], marker="o", ms=17, ls="",
                                       mfc="none", mec="black", mew=1.6, zorder=5)[0])
        info_raw.set_text(f"detections : {len(r)}\ntarget     : {ts_raw[i]:5.2f} mph\n"
                          f"controlling: {style(cr[1])[0]} @ x={cr[2]:+.0f} m" if cr
                          else f"detections : {len(r)}\ntarget     : {ts_raw[i]:5.2f} mph\n"
                               f"controlling: none")

        # ---- cleaned ----
        objs = by_frame.get(f, [])
        for tid, cls, x, y, vx, vy, observed, st, prov in objs:
            nm, col, mk = style(cls)
            # One physical object keeps ONE constant appearance: same class
            # colour, same marker, same size, whether this frame's state came
            # from a detection or from prediction.  Provenance is a diagnostic
            # overlay (--show-provenance), never a change of identity.
            artists.append(ax_cln.plot(y, x, marker=mk, color=col, ms=9, ls="",
                                       mec="black", mew=0.9, zorder=4)[0])
            if args.show_provenance and prov != tmc.PROV_OBSERVED:
                artists.append(ax_cln.plot(y, x, marker="o", ms=15, ls="",
                                           mfc="none", mec=PROV_RING[prov],
                                           mew=1.3, alpha=0.9, zorder=3)[0])
            if tid >= 0:
                artists.append(ax_cln.text(y + 1.4, x + 1.4, f"{tid}", fontsize=6.0,
                                           color="0.25", zorder=5))
                h = [(hf, hx, hy) for hf, hx, hy in history.get(tid, [])
                     if f - TRAIL <= hf <= f]
                if len(h) > 1:
                    artists.append(ax_cln.plot([p[2] for p in h], [p[1] for p in h],
                                               color=col, lw=1.1, alpha=0.55,
                                               zorder=3)[0])
            sp = np.hypot(vx, vy)
            if sp > 0.8:
                # 1 s of motion, clamped so an inflated range rate cannot draw
                # an arrow across the whole plot
                k = min(1.0, 12.0 / sp)
                artists.append(ax_cln.arrow(y, x, vy * k, vx * k, width=0.25,
                                            head_width=1.1, color=col, alpha=0.7,
                                            length_includes_head=True, zorder=3))
        cc_ = controlling_object(cln[i])
        if cc_:
            artists.append(ax_cln.plot(cc_[3], cc_[2], marker="o", ms=17, ls="",
                                       mfc="none", mec="black", mew=1.6, zorder=5)[0])
        n_coast = sum(1 for o in objs if o[8] != tmc.PROV_OBSERVED)
        info_cln.set_text(
            (f"objects    : {len(objs)}  ({n_coast} predicted)\n"
             f"target     : {ts_cln[i]:5.2f} mph\n"
             f"controlling: {style(cc_[1])[0]} @ x={cc_[2]:+.1f} m" if cc_
             else f"objects    : {len(objs)}  ({n_coast} predicted)\n"
                  f"target     : {ts_cln[i]:5.2f} mph\ncontrolling: none"))

        hdr.set_text(f"frame {f:06d}   t = {f * tmc.DT:7.2f} s   "
                     f"raw {ts_raw[i]:5.2f} mph  ->  cleaned {ts_cln[i]:5.2f} mph")
        return artists

    anim = animation.FuncAnimation(fig, draw, frames=len(idx), interval=1000 / args.fps,
                                   blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=3200,
                                    metadata={"artist": "temporal_matrix_cleaner"})
    print(f"rendering {len(idx)} frames -> {args.out} "
          f"({len(idx) / args.fps:.1f} s at {args.fps} FPS, frame step 1)")
    anim.save(str(args.out), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
