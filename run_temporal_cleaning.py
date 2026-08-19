#!/usr/bin/env python3
"""
Generate the cleaned semantic-matrix sequence from the original matrices.

The original matrices are opened read-only and never written to.  Everything
this script produces goes into a separate output directory.

    python3 run_temporal_cleaning.py
    python3 run_temporal_cleaning.py --mode causal --out cleaned_causal

Outputs (under --out):
    matrix_cleaned/NNNNNN.npy   cleaned 120x80 uint8, ORIGINAL grid/geometry
    tracks.npz                  continuous sub-cell track states per frame
                                (the "sidecar": use this if you want sub-metre
                                positions without changing the matrix grid)
    egomotion.npz               estimated ego speed / yaw rate per frame
    config.json                 the exact configuration used
    manifest.json               provenance and summary
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import temporal_matrix_cleaner as tmc


import control_params

DEFAULT_MATRIX_DIR = control_params.MATRIX_DIR
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cleaned"

GROUP_IDS = {tmc.GROUP_VRU: 0, tmc.GROUP_VEHICLE: 1,
             tmc.GROUP_SIGN: 2, tmc.GROUP_LIGHT: 3}


def load_matrices(matrix_dir: Path):
    files = sorted(matrix_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy matrices in {matrix_dir}")
    mats = []
    for p in files:
        a = np.load(p)
        if a.shape != (tmc.ROWS, tmc.COLS):
            raise ValueError(f"{p.name}: expected (120, 80), got {a.shape}")
        mats.append(a)
    return files, np.stack(mats)


PROV_IDS = {name: i for i, name in enumerate(tmc.PROVENANCES)}


def save_tracks(path: Path, result: tmc.CleanResult) -> None:
    frame, tid, cls, x, y, wx, wy, obs, static, grp, prov = ([] for _ in range(11))
    for f, ems in enumerate(result.emissions):
        for e in ems:
            frame.append(f); tid.append(e["tid"]); cls.append(e["cls"])
            x.append(e["x"]); y.append(e["y"]); wx.append(e["wx"]); wy.append(e["wy"])
            obs.append(e["observed"]); static.append(e["static"])
            grp.append(GROUP_IDS.get(e["group"], 9))
            prov.append(PROV_IDS.get(e["provenance"], 0))
    np.savez_compressed(
        path,
        frame=np.asarray(frame, dtype=np.int32),
        track_id=np.asarray(tid, dtype=np.int32),
        cls=np.asarray(cls, dtype=np.uint8),
        x_forward_m=np.asarray(x, dtype=np.float32),
        y_right_m=np.asarray(y, dtype=np.float32),
        vx_mps=np.asarray(wx, dtype=np.float32),
        vy_mps=np.asarray(wy, dtype=np.float32),
        observed=np.asarray(obs, dtype=bool),
        is_static=np.asarray(static, dtype=bool),
        group_id=np.asarray(grp, dtype=np.int8),
        group_names=np.asarray([k for k, _ in sorted(GROUP_IDS.items(),
                                                     key=lambda kv: kv[1])]),
        provenance_id=np.asarray(prov, dtype=np.int8),
        provenance_names=np.asarray(list(tmc.PROVENANCES)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    ap.add_argument("--dt", type=float, default=control_params.DT,
                    help="seconds per frame; must match the dataset rate")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--mode", choices=("offline", "causal"), default="offline",
                    help="offline uses future frames (smoothing, stitching); "
                         "causal is live-compatible")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N frames (debugging)")
    ap.add_argument("--no-ego-motion", action="store_true",
                    help="disable yaw compensation and the infrastructure prior")
    args = ap.parse_args()

    t0 = time.time()
    files, raw = load_matrices(args.matrix_dir)
    if args.limit:
        files, raw = files[:args.limit], raw[:args.limit]
    print(f"loaded {len(files)} matrices from {args.matrix_dir}")

    raw_digest = hashlib.sha256(raw.tobytes()).hexdigest()

    cfg = tmc.CleanerConfig()
    if args.no_ego_motion:
        cfg.use_ego_motion = False

    print(f"cleaning in '{args.mode}' mode ...")
    result = tmc.clean_sequence(raw, cfg=cfg, mode=args.mode)

    # The original data must be provably untouched.
    if hashlib.sha256(raw.tobytes()).hexdigest() != raw_digest:
        raise RuntimeError("input matrices were modified in memory -- aborting")

    out = args.out
    mat_out = out / "matrix_cleaned"
    mat_out.mkdir(parents=True, exist_ok=True)
    for p, m in zip(files, result.cleaned):
        np.save(mat_out / p.name, m)

    save_tracks(out / "tracks.npz", result)
    np.savez_compressed(out / "egomotion.npz",
                        ego_speed_mps=result.ego_v.astype(np.float32),
                        ego_yaw_rate_radps=result.ego_psi.astype(np.float32),
                        n_inliers=result.ego_inliers.astype(np.int16))
    (out / "config.json").write_text(cfg.to_json())

    if abs(args.dt - tmc.DT) > 1e-9:
        raise ValueError(f"--dt {args.dt} does not match the cleaner's DT {tmc.DT}")
    n_emit = sum(len(e) for e in result.emissions)
    prov_counts = {}
    for ems in result.emissions:
        for e in ems:
            prov_counts[e["provenance"]] = prov_counts.get(e["provenance"], 0) + 1
    manifest = {
        "source_matrix_dir": str(args.matrix_dir.resolve()),
        "source_frames": len(files),
        "source_sha256": raw_digest,
        "first_frame": files[0].name,
        "last_frame": files[-1].name,
        "mode": args.mode,
        "uses_future_frames": args.mode == "offline",
        "fps": 1.0 / tmc.DT,
        "dt_s": tmc.DT,
        "duration_s": round(len(files) * tmc.DT, 2),
        "grid": {"rows": tmc.ROWS, "cols": tmc.COLS,
                 "cell_size_m": tmc.CELL_SIZE_M,
                 "ego_row": tmc.EGO_ROW, "ego_col": tmc.EGO_COL},
        "n_tracks": len(result.tracks),
        "n_emitted_objects": n_emit,
        "raw_nonzero_cells": int((raw != 0).sum()),
        "cleaned_nonzero_cells": int((result.cleaned != 0).sum()),
        "provenance_counts": prov_counts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {len(files)} cleaned matrices to {mat_out}")
    print(f"      {out/'tracks.npz'}  ({n_emit} object-frames, {len(result.tracks)} tracks)")
    print(f"      {out/'egomotion.npz'}, {out/'config.json'}, {out/'manifest.json'}")
    print(f"original matrices untouched (sha256 {raw_digest[:16]}...)")
    print(f"object-state provenance: " +
          ", ".join(f"{k}={v}" for k, v in sorted(prov_counts.items())))
    print(f"done in {manifest['elapsed_s']} s")


if __name__ == "__main__":
    main()
