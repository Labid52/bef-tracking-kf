#!/usr/bin/env python3
"""
Four-way BEV comparison using four synchronized matrix folders:

RAW      : detector / distance-estimation output
OFFLINE  : offline cleaner output
ONLINE   : causal cleaner rerun on recorded data
REALTIME : matrix saved from the actual live/Thor implementation

All four panels use the same matrix-only visualization so the comparison is fair.
No velocity arrows or track-sidecar data are used.

Example
-------
python3 compare_four_bev_modes.py \
  --raw-dir capture/raw \
  --offline-dir capture/offline \
  --online-dir capture/online \
  --realtime-dir capture/realtime \
  --out capture/raw_offline_online_realtime.mp4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

try:
    import control_params
    FPS_DEFAULT = int(control_params.FPS)
    TS_KW = dict(control_params.TARGET_SPEED_KW)
except Exception:
    FPS_DEFAULT = 10
    TS_KW = dict(
        width_of_interest_m=4.0,
        ego_speed_mph=20.0,
        safety_stopping_distance_m=8.0,
        acceleration_mps2=1.1176,
        deceleration_mps2=1.1176,
        dt=0.10,
        max_target_speed_mph=20.0,
        hood_ignore_distance_m=6.0,
        stop_offset_m=5.0,
        stop_lateral_limit_m=15.0,
    )

from target_speed import getTargetSpeed

ROWS, COLS = 120, 80
EGO_ROW, EGO_COL = 80, 40
X_MIN, X_MAX = -39.0, 80.0
Y_MIN, Y_MAX = -40.0, 39.0

TS_OBSTACLE_CLASSES = {1, 2, 3, 4, 5, 6}
TS_STOP_CLASSES = {7, 9}

# Fixed class appearance. Colors never depend on frame content or order.
CLASS_STYLE = {
    1: ("person",        "#e8453c", "o"),
    2: ("bicycle",       "#e8853c", "o"),
    3: ("car",           "#3c8ce8", "s"),
    4: ("motorcycle",    "#8c3ce8", "s"),
    5: ("bus",           "#00a0a0", "s"),
    6: ("truck",         "#1f5fa0", "s"),
    7: ("stop sign",     "#d40000", "H"),
    8: ("traffic light", "#888888", "^"),
    9: ("red light",     "#d40000", "^"),
    10: ("yellow light", "#d0a000", "^"),
    11: ("green light",  "#22a022", "^"),
    255: ("unknown",     "#999999", "x"),
}
DEFAULT_STYLE = ("unknown", "#999999", "x")


def style(cls):
    return CLASS_STYLE.get(int(cls), DEFAULT_STYLE)


def controlling_object(matrix):
    """Return (margin, cls, x, y) for the object limiting target speed."""
    best = None
    rr, cc = np.nonzero(matrix)
    for r, c in zip(rr.tolist(), cc.tolist()):
        cls = int(matrix[r, c])
        x = float(EGO_ROW - r)
        y = float(c - EGO_COL)

        if x <= 0:
            continue

        if cls in TS_OBSTACLE_CLASSES and abs(y) <= TS_KW["width_of_interest_m"] / 2:
            if cls == 3 and x <= TS_KW["hood_ignore_distance_m"]:
                continue
            margin = x - TS_KW["safety_stopping_distance_m"]
        elif cls in TS_STOP_CLASSES and abs(y) <= TS_KW["stop_lateral_limit_m"]:
            margin = x - TS_KW["stop_offset_m"]
        else:
            continue

        if best is None or margin < best[0]:
            best = (margin, cls, x, y)

    return best


def setup_axis(ax, title):
    ax.set_xlim(Y_MIN - 1, Y_MAX + 1)
    ax.set_ylim(X_MIN - 1, X_MAX + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral y [m] (right positive)", fontsize=8)
    ax.set_ylabel("forward x [m]", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8, ls=":")

    # Same policy regions as the original validation animation.
    hw = TS_KW["width_of_interest_m"] / 2
    ax.add_patch(Rectangle(
        (-hw, 0), 2 * hw, X_MAX,
        color="#3c8ce8", alpha=0.07, zorder=0
    ))
    sl = TS_KW["stop_lateral_limit_m"]
    ax.add_patch(Rectangle(
        (-sl, 0), 2 * sl, X_MAX,
        color="#d40000", alpha=0.035, zorder=0
    ))

    ax.plot(0, 0, marker="^", ms=12, color="black", zorder=6)
    ax.text(0, -6, "ego", ha="center", fontsize=7)


def list_npy(folder):
    files = sorted(Path(folder).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {folder}")
    return files


def nearest_same_group_stats(raw, other):
    """
    Simple frame diagnostic only:
    for every nonzero output point, distance to nearest raw point in same
    semantic group. This is not tracking ground truth.
    """
    def group(cls):
        cls = int(cls)
        if cls in (1, 2):
            return "vru"
        if cls in (3, 4, 5, 6):
            return "vehicle"
        if cls == 7:
            return "sign"
        if cls in (8, 9, 10, 11):
            return "light"
        return "other"

    rr, rc = np.nonzero(raw)
    orr, occ = np.nonzero(other)
    if len(orr) == 0:
        return 0, 0, 0.0
    if len(rr) == 0:
        return len(orr), len(orr), float("nan")

    raw_by_g = {}
    for r, c in zip(rr.tolist(), rc.tolist()):
        g = group(raw[r, c])
        raw_by_g.setdefault(g, []).append((EGO_ROW-r, c-EGO_COL))

    ds = []
    for r, c in zip(orr.tolist(), occ.tolist()):
        g = group(other[r, c])
        candidates = raw_by_g.get(g, [])
        if not candidates:
            ds.append(float("inf"))
            continue
        p = np.array([EGO_ROW-r, c-EGO_COL], dtype=float)
        q = np.asarray(candidates, dtype=float)
        ds.append(float(np.min(np.linalg.norm(q-p, axis=1))))

    ds = np.asarray(ds, dtype=float)
    finite = ds[np.isfinite(ds)]
    mean_d = float(np.mean(finite)) if len(finite) else float("nan")
    return int(np.sum(ds > 3.0)), int(np.sum(ds > 6.0)), mean_d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--offline-dir", type=Path)
    ap.add_argument("--online-dir", type=Path)
    ap.add_argument("--realtime-dir", type=Path)
    ap.add_argument("--panel", action="append", default=None, metavar="NAME=DIR",
                    help="explicit panel, repeatable, e.g. "
                         "--panel 'FIXED ONLINE=out/matrix_cleaned'. "
                         "Overrides the four --*-dir options. 2-6 panels.")
    ap.add_argument("--out", type=Path, default=Path("raw_offline_online_realtime.mp4"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--dpi", type=int, default=105)
    args = ap.parse_args()

    if args.panel:
        folders = {}
        for spec in args.panel:
            if "=" not in spec:
                raise SystemExit(f"--panel needs NAME=DIR, got {spec!r}")
            name, d = spec.split("=", 1)
            folders[name.strip()] = Path(d)
        if not (2 <= len(folders) <= 6):
            raise SystemExit("give between 2 and 6 panels")
    else:
        need = dict(raw_dir=args.raw_dir, offline_dir=args.offline_dir,
                    online_dir=args.online_dir, realtime_dir=args.realtime_dir)
        missing = [k for k, v in need.items() if v is None]
        if missing:
            raise SystemExit(f"missing {missing}; or use --panel NAME=DIR")
        folders = {
            "RAW DETECTION": args.raw_dir,
            "OFFLINE": args.offline_dir,
            "ONLINE / CAUSAL": args.online_dir,
            "REALTIME / THOR": args.realtime_dir,
        }
    files = {name: list_npy(folder) for name, folder in folders.items()}

    counts = {name: len(v) for name, v in files.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Frame counts differ: {counts}")

    n = next(iter(counts.values()))
    lo = max(0, args.start)
    hi = n if args.end is None else min(args.end, n)
    idx = list(range(lo, hi))
    if not idx:
        raise ValueError("Empty frame range")

    # Load only requested range.
    data = {
        name: np.stack([np.load(flist[i]) for i in idx])
        for name, flist in files.items()
    }
    for name, arr in data.items():
        if arr.shape[1:] != (ROWS, COLS):
            raise ValueError(f"{name}: expected 120x80 matrices, got {arr.shape[1:]}")

    ts = {
        name: np.asarray([getTargetSpeed(matrix=m, **TS_KW) for m in arr])
        for name, arr in data.items()
    }

    names = list(folders)
    ncol = 2 if len(names) <= 4 else 3
    nrow = (len(names) + ncol - 1) // ncol
    fig = plt.figure(figsize=(7.1 * ncol, 4.7 * nrow + 1.9))
    gs = fig.add_gridspec(
        nrow + 1, ncol,
        height_ratios=[2.35] * nrow + [1.15],
        hspace=0.30, wspace=0.16,
        left=0.055, right=0.985, top=0.945, bottom=0.10
    )
    axes = {nm: fig.add_subplot(gs[i // ncol, i % ncol])
            for i, nm in enumerate(names)}
    for name, ax in axes.items():
        setup_axis(ax, name)

    ax_ts = fig.add_subplot(gs[nrow, :])
    ax_ts.set_xlim(idx[0], idx[-1])
    ax_ts.set_ylim(-1, 21.5)
    ax_ts.set_xlabel("frame")
    ax_ts.set_ylabel("target speed [mph]")
    ax_ts.grid(alpha=0.25, linewidth=0.5)

    # Fixed colours, assigned by panel order for any panel set.  No blinking:
    # a panel keeps the same colour for the whole video.
    curve_style = {
        "RAW DETECTION": ("#777777", 1.0),
        "OFFLINE": ("#2ca02c", 1.3),
        "ONLINE / CAUSAL": ("#1f77b4", 1.3),
        "REALTIME / THOR": ("#d62728", 1.5),
    }
    palette = [("#777777", 1.0), ("#d62728", 1.3), ("#1f77b4", 1.5),
               ("#2ca02c", 1.3), ("#9467bd", 1.3), ("#8c564b", 1.3)]
    for i, name in enumerate(folders):
        color, lw = curve_style.get(name, palette[i % len(palette)])
        ax_ts.plot(idx, ts[name], color=color, lw=lw, label=name)
    ax_ts.legend(loc="lower left", fontsize=8, ncol=4)
    cursor = ax_ts.axvline(idx[0], color="black", lw=1.0)

    hdr = fig.text(
        0.5, 0.99, "", ha="center", va="top",
        fontsize=11, family="monospace"
    )

    info = {}
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9)
    for name, ax in axes.items():
        info[name] = ax.text(
            0.02, 0.985, "", transform=ax.transAxes,
            va="top", fontsize=7.6, family="monospace", bbox=box
        )

    handles = [
        plt.Line2D([], [], color=c, marker=mk, ls="", ms=6, label=nm)
        for nm, c, mk in CLASS_STYLE.values()
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        fontsize=7, framealpha=0.9, ncol=6
    )

    artists = []

    def clear_dynamic():
        while artists:
            a = artists.pop()
            try:
                a.remove()
            except Exception:
                pass

    raw_name = "RAW DETECTION"

    def draw(k):
        clear_dynamic()
        f = idx[k]
        cursor.set_xdata([f, f])

        for name, ax in axes.items():
            m = data[name][k]
            rr, cc = np.nonzero(m)

            for r, c in zip(rr.tolist(), cc.tolist()):
                cls = int(m[r, c])
                _, col, mk = style(cls)
                artists.append(
                    ax.plot(
                        c-EGO_COL, EGO_ROW-r,
                        marker=mk, color=col, ms=7.5, ls="",
                        mec="black", mew=0.4, zorder=4
                    )[0]
                )

            ctrl = controlling_object(m)
            if ctrl is not None:
                artists.append(
                    ax.plot(
                        ctrl[3], ctrl[2],
                        marker="o", ms=15, ls="",
                        mfc="none", mec="black", mew=1.5, zorder=5
                    )[0]
                )

            if name == raw_name:
                distance_line = "raw reference"
            else:
                n3, n6, mean_d = nearest_same_group_stats(data[raw_name][k], m)
                md = "n/a" if np.isnan(mean_d) else f"{mean_d:.1f}m"
                distance_line = f">3m:{n3:2d} >6m:{n6:2d} mean:{md}"

            ctext = (
                f"{style(ctrl[1])[0]} @ x={ctrl[2]:+.0f}m"
                if ctrl is not None else "none"
            )
            info[name].set_text(
                f"points : {len(rr):3d}\n"
                f"target : {ts[name][k]:5.2f} mph\n"
                f"control: {ctext}\n"
                f"{distance_line}"
            )

        hdr.set_text(
            f"frame {f:06d}   t={f/args.fps:7.2f}s   |   target speed [mph]:  "
            + "   ".join(f"{nm} {ts[nm][k]:5.2f}" for nm in folders)
        )
        return artists

    anim = animation.FuncAnimation(
        fig, draw, frames=len(idx),
        interval=1000.0/args.fps, blit=False
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=4000,
        metadata={"artist": "four-way BEV diagnostic"}
    )
    print(
        f"Rendering {len(idx)} frames -> {args.out} "
        f"({len(idx)/args.fps:.1f}s at {args.fps} FPS)"
    )
    anim.save(str(args.out), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
