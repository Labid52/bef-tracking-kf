#!/usr/bin/env python3
"""
Inspect the 120x80 semantic matrix geometry and metadata.

Run from:
    /home/labid/mila/longitudinal_control

Example:
    python3 inspect_matrix_geometry.py

Optional:
    python3 inspect_matrix_geometry.py --matrix-dir matrix --check-all
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def first_number(text: str) -> float:
    """Extract the first signed number from a metadata string."""
    m = re.search(r"[-+]?\d*\.?\d+", str(text))
    if not m:
        raise ValueError(f"Could not find a number in: {text!r}")
    return float(m.group())


def load_metadata(matrix_dir: Path):
    path = matrix_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r") as f:
        return json.load(f)


def derive_geometry(meta):
    rows, cols = map(int, meta["shape_rows_columns"])
    cell = float(meta["cell_size_m"])

    # Metadata meanings:
    # row 0        -> forward +80 m
    # last row     -> rear -40 m
    # column 0     -> left -40 m
    # last column  -> right +40 m
    front_m = first_number(meta["row_0"])
    rear_m = first_number(meta["last_row"])
    left_m = first_number(meta["column_0"])
    right_m = first_number(meta["last_column"])

    # For a raster covering [rear, front] longitudinally and [left, right]
    # laterally, x=0 and y=0 fall at these nominal grid indices.
    ego_row = int(round(front_m / cell))
    ego_col = int(round((0.0 - left_m) / cell))

    return {
        "rows": rows,
        "cols": cols,
        "cell": cell,
        "front_m": front_m,
        "rear_m": rear_m,
        "left_m": left_m,
        "right_m": right_m,
        "ego_row": ego_row,
        "ego_col": ego_col,
    }


def cell_to_metric(row, col, geom, use_cell_center=True):
    """
    Convert a matrix cell index to approximate ego-relative metric coordinates.

    x_forward_m > 0 : ahead of ego
    x_forward_m < 0 : behind ego

    y_right_m > 0   : right of ego
    y_right_m < 0   : left of ego

    Because metadata gives physical grid limits, cell-center coordinates
    are offset by 0.5 cell.
    """
    cell = geom["cell"]
    if use_cell_center:
        x_forward_m = geom["front_m"] - (row + 0.5) * cell
        y_right_m = geom["left_m"] + (col + 0.5) * cell
    else:
        x_forward_m = geom["front_m"] - row * cell
        y_right_m = geom["left_m"] + col * cell
    return x_forward_m, y_right_m


def inspect_one_matrix(path: Path, meta, geom):
    a = np.load(path)

    print("\n" + "=" * 72)
    print(f"FRAME TEST: {path.name}")
    print("=" * 72)
    print(f"shape : {a.shape}")
    print(f"dtype : {a.dtype}")
    print(f"unique values: {np.unique(a).tolist()}")

    if tuple(a.shape) != (geom["rows"], geom["cols"]):
        print("WARNING: matrix shape does not match metadata.")

    class_ids = {int(k): v for k, v in meta["class_ids"].items()}

    rr, cc = np.where(a != 0)
    print(f"non-empty cells: {len(rr)}")

    if len(rr) == 0:
        print("No objects/features in this frame.")
        return

    print("\nNon-empty cells:")
    for r, c in zip(rr, cc):
        cid = int(a[r, c])
        name = class_ids.get(cid, "UNMAPPED")
        x_m, y_m = cell_to_metric(r, c, geom, use_cell_center=True)
        print(
            f"  row={r:3d}, col={c:2d}, class={cid:3d} ({name:13s}), "
            f"x_forward≈{x_m:6.1f} m, y_right≈{y_m:6.1f} m"
        )


def scan_dataset(matrix_dir: Path, meta, geom, check_all=False):
    files = sorted(matrix_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {matrix_dir}")

    # If not doing a full integrity test, we still scan all frames for
    # lightweight class/position statistics.
    class_counts = Counter()
    car_rows = []
    shapes = Counter()
    dtypes = Counter()

    for i, path in enumerate(files):
        a = np.load(path, mmap_mode="r")
        shapes[tuple(a.shape)] += 1
        dtypes[str(a.dtype)] += 1

        vals, counts = np.unique(a, return_counts=True)
        for v, n in zip(vals, counts):
            if int(v) != 0:
                class_counts[int(v)] += int(n)

        car_r, _ = np.where(a == 3)
        car_rows.extend(car_r.tolist())

        if check_all and tuple(a.shape) != (geom["rows"], geom["cols"]):
            print(f"BAD SHAPE: {path.name}: {a.shape}")

    return files, class_counts, np.asarray(car_rows), shapes, dtypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-dir",
        default="/home/labid/mila/longitudinal_control/matrix",
        help="Directory containing metadata.json and the .npy matrices",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="Specific frame filename, e.g. 000500.npy",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Perform full shape consistency reporting",
    )
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir)
    meta = load_metadata(matrix_dir)
    geom = derive_geometry(meta)

    class_ids = {int(k): v for k, v in meta["class_ids"].items()}

    print("=" * 72)
    print("METADATA")
    print("=" * 72)
    print(json.dumps(meta, indent=2))

    print("\n" + "=" * 72)
    print("ANSWERS TO THE FOUR GEOMETRY QUESTIONS")
    print("=" * 72)

    print("\n1) WHERE IS THE EGO VEHICLE?")
    print("   The ego is not stored as a semantic class in the matrix.")
    print("   The matrix is ego-relative, so ego is the physical origin (0 m, 0 m).")
    print(
        f"   Derived nominal grid origin: row={geom['ego_row']}, "
        f"col={geom['ego_col']}."
    )
    print("   For this metadata, that is approximately matrix index (80, 40).")
    print(
        "   IMPORTANT: because this is a 1 m raster, the exact physical origin "
        "lies on a cell boundary unless the producer used a different center convention."
    )

    print("\n2) WHICH MATRIX DIRECTION IS FORWARD?")
    print(f"   row 0 metadata : {meta['row_0']}")
    print(f"   last-row metadata: {meta['last_row']}")
    print("   Therefore: SMALLER row index = farther FORWARD.")
    print("              LARGER row index  = farther REARWARD.")
    print(f"   Ego longitudinal boundary is near row {geom['ego_row']}.")

    print("\n   Lateral direction:")
    print(f"   column 0 metadata : {meta['column_0']}")
    print(f"   last-column metadata: {meta['last_column']}")
    print("   Therefore: smaller column = LEFT, larger column = RIGHT.")
    print(f"   Ego centerline is near column {geom['ego_col']}.")

    print("\n3) WHAT IS METERS PER CELL?")
    print(f"   metadata cell_size_m = {geom['cell']} m/cell")
    print(
        f"   matrix shape = {geom['rows']} x {geom['cols']} -> "
        f"{geom['rows'] * geom['cell']:.1f} m longitudinal x "
        f"{geom['cols'] * geom['cell']:.1f} m lateral"
    )
    print(
        f"   physical extent = rear {geom['rear_m']:+.1f} m to "
        f"front {geom['front_m']:+.1f} m, "
        f"left {geom['left_m']:+.1f} m to right {geom['right_m']:+.1f} m"
    )

    print("\n4) WHAT DO THE CLASS VALUES MEAN?")
    for cid in sorted(class_ids):
        print(f"   {cid:3d} -> {class_ids[cid]}")

    files, class_counts, car_rows, shapes, dtypes = scan_dataset(
        matrix_dir, meta, geom, check_all=args.check_all
    )

    print("\n" + "=" * 72)
    print("DATASET SANITY CHECK")
    print("=" * 72)
    print(f"number of matrices: {len(files)}")
    print(f"observed shapes: {dict(shapes)}")
    print(f"observed dtypes: {dict(dtypes)}")

    print("\nNon-empty class-cell counts:")
    for cid, count in sorted(class_counts.items()):
        print(f"   {cid:3d} ({class_ids.get(cid, 'UNMAPPED'):13s}) : {count}")

    if car_rows.size:
        ahead = np.sum(car_rows < geom["ego_row"])
        at_or_behind = np.sum(car_rows >= geom["ego_row"])
        pct = 100.0 * ahead / car_rows.size
        print("\nClass-3 (car) positional sanity check:")
        print(f"   total car cells            : {car_rows.size}")
        print(f"   rows < ego row ({geom['ego_row']}) : {ahead} ({pct:.1f}%)")
        print(f"   rows >= ego row            : {at_or_behind}")
        print(
            "   If most driving-scene cars are expected ahead, this supports "
            "the metadata interpretation that decreasing row means forward."
        )

    # Inspect one example frame.
    if args.frame:
        frame_path = matrix_dir / args.frame
        if not frame_path.exists():
            raise FileNotFoundError(frame_path)
    else:
        # Pick the frame with the most class-3 cells so the example is informative.
        best_path = None
        best_n = -1
        for path in files:
            a = np.load(path, mmap_mode="r")
            n = int(np.count_nonzero(a == 3))
            if n > best_n:
                best_n = n
                best_path = path
        frame_path = best_path

    inspect_one_matrix(frame_path, meta, geom)

    print("\n" + "=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print("Nominal ego origin : (row 80, col 40)")
    print("Forward direction  : toward smaller row indices")
    print("Lateral direction  : smaller col = left, larger col = right")
    print(f"Resolution         : {geom['cell']} m/cell")
    print("Class 3            : car")
    print(
        "\nFor target-speed work, use row < 80 for objects ahead and convert "
        "distance relative to the ego origin. Before final vehicle control, "
        "check the matrix-generation code once to remove the remaining "
        "half-cell/boundary ambiguity."
    )


if __name__ == "__main__":
    main()
