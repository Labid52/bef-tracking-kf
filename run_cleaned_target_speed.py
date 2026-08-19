#!/usr/bin/env python3
"""
Streaming demo: raw matrix -> OnlineTemporalCleaner.update() -> getTargetSpeed().

This is the LIVE/CAUSAL path.  It processes one frame at a time and never looks
at a future matrix, so it is what a real 10 FPS pipeline would call.  Use
run_temporal_cleaning.py instead when you want the higher-quality OFFLINE
result written to disk.

HISTORY: this script previously imported ``TemporalMatrixCleaner`` from an
earlier implementation that no longer exists, pointed at a different checkout
of the project, and passed dt = 0.05 to a 10 FPS dataset.  All three are fixed;
the streaming call shape (``cleaner.update(matrix, dt=...)``) is unchanged, and
it now returns the emitted object list alongside the cleaned matrix.

    python3 run_cleaned_target_speed.py
    python3 run_cleaned_target_speed.py --start 1140 --end 1180
    python3 run_cleaned_target_speed.py --csv cleaned_causal/target_speed_stream.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np

import control_params as cp
from temporal_matrix_cleaner import OnlineTemporalCleaner
from target_speed import getTargetSpeed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, default=cp.MATRIX_DIR)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write frame,target_speed_raw,target_speed_cleaned")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    files = sorted(args.matrix_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy matrices in {args.matrix_dir}")
    files = files[args.start:args.end]

    cleaner = OnlineTemporalCleaner()
    rows = []
    latency_ms = []

    if not args.quiet:
        print(f"{'frame':<14} {'raw_mph':>9} {'cleaned_mph':>13} {'objects':>8} "
              f"{'v_ego':>7} {'yaw':>7}")
        print("-" * 64)

    for path in files:
        raw = np.load(path)

        t0 = time.perf_counter()
        # --- the entire live interface is this one call -------------------
        cleaned, objects = cleaner.update(raw, dt=cp.DT)
        # If the vehicle publishes a yaw rate, pass it in and the internal
        # estimator is bypassed:
        #     cleaner.update(raw, dt=cp.DT, ego_yaw_rate=imu_yaw_rate)
        latency_ms.append((time.perf_counter() - t0) * 1e3)

        ts_raw = getTargetSpeed(matrix=raw, **cp.TARGET_SPEED_KW)
        ts_cln = getTargetSpeed(matrix=cleaned, **cp.TARGET_SPEED_KW)
        rows.append((path.stem, ts_raw, ts_cln))

        if not args.quiet:
            print(f"{path.name:<14} {ts_raw:>9.2f} {ts_cln:>13.2f} {len(objects):>8} "
                  f"{cleaner.ego_speed_mps:>7.2f} {cleaner.ego_yaw_rate:>+7.3f}")

    lat = np.asarray(latency_ms)
    r = np.array([x[1] for x in rows])
    c = np.array([x[2] for x in rows])
    print()
    print(f"frames                       : {len(rows)}  ({len(rows) * cp.DT:.1f} s at {cp.FPS:.0f} FPS)")
    print(f"mean |d target speed|/frame  : raw {np.abs(np.diff(r)).mean():.4f} mph"
          f"   cleaned {np.abs(np.diff(c)).mean():.4f} mph")
    print(f"changes > 10 mph             : raw {int((np.abs(np.diff(r)) > 10).sum())}"
          f"   cleaned {int((np.abs(np.diff(c)) > 10).sum())}")
    print(f"update() latency             : mean {lat.mean():.2f} ms, "
          f"p99 {np.percentile(lat, 99):.2f} ms, max {lat.max():.2f} ms "
          f"(budget {cp.DT * 1e3:.0f} ms)")
    print(f"safety pass-through restores : {cleaner.n_restored}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as fh:
            fh.write("frame,target_speed_raw_mph,target_speed_cleaned_mph\n")
            for name, a, b in rows:
                fh.write(f"{name},{a:.4f},{b:.4f}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
