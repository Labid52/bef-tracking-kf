#!/usr/bin/env python3
"""Independent correctness oracle for the accepted longitudinal safety V1.

Expected boundaries, zones, and targets are computed locally with basic
``math`` operations. Production helpers are called only for actual results.
This validates transformations, not the physical truth of BEV detections.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import zipfile

import numpy as np

from bev_longitudinal_adapter import extract_bev_longitudinal_observation
from gps_sensor_adapter import load_gps_observations_from_archive
from live_bev_viewer import LiveBEVProcessor
from longitudinal_safety_core import (
    LongitudinalSafetyState,
    LongitudinalUncertainty,
    evaluate_longitudinal_state,
)


MPH_TO_MPS = 0.44704
NOMINAL_TARGET_MPH = 20.0
STANDSTILL_CLEARANCE_M = 2.0
MACHINE_RESPONSE_DELAY_S = 0.30
CRITICAL_EXTRA_MARGIN_S = 0.45
COMFORTABLE_DECELERATION_MPS2 = 2.0
EMERGENCY_DECELERATION_MPS2 = 9.0
ABS_TOL = 1e-12
BOUNDARY_EPSILON_M = 1e-9
PHYSICAL_CLASSES = frozenset({1, 2, 3, 4, 5, 6})
ZONE_RANK = {"SAFE": 0, "CRITICAL": 1, "COLLISION": 2}


@dataclass(frozen=True)
class OracleResult:
    collision_boundary_m: float
    critical_boundary_m: float
    zone: str
    safe_target_speed_mph: float


def oracle_evaluate(
    ego_speed_mps: float,
    obstacle_gap_m: Optional[float],
    nominal_target_speed_mph: float,
    *,
    speed_uncertainty_mps: float = 0.0,
    gap_uncertainty_m: float = 0.0,
    delay_uncertainty_s: float = 0.0,
) -> OracleResult:
    """Independent literal implementation of the accepted equations."""
    effective_speed_mps = ego_speed_mps + speed_uncertainty_mps
    effective_delay_s = MACHINE_RESPONSE_DELAY_S + delay_uncertainty_s
    effective_gap_m = (
        None if obstacle_gap_m is None
        else max(0.0, obstacle_gap_m - gap_uncertainty_m)
    )
    collision_m = (
        STANDSTILL_CLEARANCE_M
        + effective_speed_mps * effective_delay_s
        + effective_speed_mps * effective_speed_mps
        / (2.0 * EMERGENCY_DECELERATION_MPS2)
    )
    critical_m = (
        STANDSTILL_CLEARANCE_M
        + effective_speed_mps
        * (effective_delay_s + CRITICAL_EXTRA_MARGIN_S)
        + effective_speed_mps * effective_speed_mps
        / (2.0 * COMFORTABLE_DECELERATION_MPS2)
    )
    if effective_gap_m is None or effective_gap_m > critical_m:
        zone = "SAFE"
        safe_mph = nominal_target_speed_mph
    elif effective_gap_m <= collision_m:
        zone = "COLLISION"
        safe_mph = 0.0
    else:
        zone = "CRITICAL"
        fraction = (
            (effective_gap_m - collision_m) / (critical_m - collision_m)
        )
        safe_mph = nominal_target_speed_mph * math.sqrt(fraction)
    safe_mph = min(nominal_target_speed_mph, max(0.0, safe_mph))
    return OracleResult(collision_m, critical_m, zone, safe_mph)


def manual_bev_range_m(matrix: np.ndarray) -> Optional[float]:
    """Independent default-adapter oracle using documented raster geometry."""
    candidates: list[float] = []
    for row in range(120):
        for col in range(80):
            if int(matrix[row, col]) not in PHYSICAL_CLASSES:
                continue
            x_forward_m = float(80 - row)
            y_right_m = float(col - 40)
            if x_forward_m > 0.0 and abs(y_right_m) <= 2.0:
                candidates.append(x_forward_m)
    return min(candidates) if candidates else None


def close(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=ABS_TOL)


def csv_optional(row: dict[str, str], key: str) -> Optional[float]:
    value = row[key]
    return None if value in ("", "None") else float(value)


def validate_grid(output_csv: Path) -> dict[str, object]:
    fields = (
        "speed_mph", "gap_m", "oracle_collision_m", "production_collision_m",
        "oracle_critical_m", "production_critical_m", "oracle_zone",
        "production_zone", "oracle_safe_target_mph", "production_safe_target_mph",
    )
    rows = []
    boundary_mismatches = zone_mismatches = target_mismatches = 0
    maximum_boundary_error_m = maximum_target_error_mph = 0.0
    for speed_mph in range(41):
        speed_mps = speed_mph * MPH_TO_MPS
        for gap_m in range(81):
            expected = oracle_evaluate(speed_mps, float(gap_m), NOMINAL_TARGET_MPH)
            actual = evaluate_longitudinal_state(
                LongitudinalSafetyState(
                    speed_mps, float(gap_m), NOMINAL_TARGET_MPH, 80.0
                )
            )
            collision_error = abs(
                expected.collision_boundary_m - actual.collision_boundary_m
            )
            critical_error = abs(
                expected.critical_boundary_m - actual.critical_boundary_m
            )
            target_error = abs(
                expected.safe_target_speed_mph - actual.safe_target_speed_mph
            )
            maximum_boundary_error_m = max(
                maximum_boundary_error_m, collision_error, critical_error
            )
            maximum_target_error_mph = max(maximum_target_error_mph, target_error)
            if collision_error > ABS_TOL or critical_error > ABS_TOL:
                boundary_mismatches += 1
            if expected.zone != actual.zone:
                zone_mismatches += 1
            if target_error > ABS_TOL:
                target_mismatches += 1
            rows.append({
                "speed_mph": speed_mph,
                "gap_m": gap_m,
                "oracle_collision_m": expected.collision_boundary_m,
                "production_collision_m": actual.collision_boundary_m,
                "oracle_critical_m": expected.critical_boundary_m,
                "production_critical_m": actual.critical_boundary_m,
                "oracle_zone": expected.zone,
                "production_zone": actual.zone,
                "oracle_safe_target_mph": expected.safe_target_speed_mph,
                "production_safe_target_mph": actual.safe_target_speed_mph,
            })
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "states": len(rows),
        "boundary_mismatches": boundary_mismatches,
        "zone_mismatches": zone_mismatches,
        "target_mismatches": target_mismatches,
        "maximum_boundary_error_m": maximum_boundary_error_m,
        "maximum_target_error_mph": maximum_target_error_mph,
    }


def validate_boundaries_and_semantics() -> tuple[list[tuple], int, bool]:
    table = []
    previous = None
    monotonic = True
    for mph in range(0, 41, 5):
        expected = oracle_evaluate(mph * MPH_TO_MPS, None, NOMINAL_TARGET_MPH)
        table.append((mph, expected.collision_boundary_m, expected.critical_boundary_m))
        if previous is not None:
            monotonic &= (
                expected.collision_boundary_m >= previous[0]
                and expected.critical_boundary_m >= previous[1]
            )
        previous = (expected.collision_boundary_m, expected.critical_boundary_m)

    mismatches = 0
    for mph in (5, 10, 20, 30, 40):
        expected = oracle_evaluate(mph * MPH_TO_MPS, None, NOMINAL_TARGET_MPH)
        probes = (
            (expected.collision_boundary_m - BOUNDARY_EPSILON_M, "COLLISION"),
            (expected.collision_boundary_m, "COLLISION"),
            (expected.collision_boundary_m + BOUNDARY_EPSILON_M, "CRITICAL"),
            (expected.critical_boundary_m - BOUNDARY_EPSILON_M, "CRITICAL"),
            (expected.critical_boundary_m, "CRITICAL"),
            (expected.critical_boundary_m + BOUNDARY_EPSILON_M, "SAFE"),
        )
        for gap_m, expected_zone in probes:
            actual = evaluate_longitudinal_state(
                LongitudinalSafetyState(
                    mph * MPH_TO_MPS, gap_m, NOMINAL_TARGET_MPH
                )
            )
            mismatches += actual.zone != expected_zone
    return table, mismatches, monotonic


def fixed_gap_sampled_zones() -> list[tuple[int, str]]:
    rows = []
    for gap_m in (5, 10, 15, 20, 30):
        compact = []
        prior_zone = None
        start_mph = 0
        for mph in range(41):
            zone = oracle_evaluate(
                mph * MPH_TO_MPS, float(gap_m), NOMINAL_TARGET_MPH
            ).zone
            if prior_zone is None:
                prior_zone = zone
            elif zone != prior_zone:
                compact.append(f"{start_mph}-{mph - 1} mph {prior_zone}")
                start_mph, prior_zone = mph, zone
        compact.append(f"{start_mph}-40 mph {prior_zone}")
        rows.append((gap_m, "; ".join(compact)))
    return rows


def exact_gap_thresholds() -> list[tuple]:
    rows = []
    for gap_m in (5.0, 10.0, 20.0, 30.0):
        critical_mps = positive_boundary_root_mps(
            gap_m, MACHINE_RESPONSE_DELAY_S + CRITICAL_EXTRA_MARGIN_S,
            COMFORTABLE_DECELERATION_MPS2,
        )
        collision_mps = positive_boundary_root_mps(
            gap_m, MACHINE_RESPONSE_DELAY_S, EMERGENCY_DECELERATION_MPS2,
        )
        rows.append((gap_m, critical_mps / MPH_TO_MPS, collision_mps / MPH_TO_MPS))
    return rows


def positive_boundary_root_mps(gap_m: float, delay_s: float, decel: float) -> float:
    available_m = gap_m - STANDSTILL_CLEARANCE_M
    return decel * (math.sqrt(delay_s * delay_s + 2.0 * available_m / decel) - delay_s)


def validate_recorded(
    archive_path: Path,
    replay_csv_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    observations = load_gps_observations_from_archive(archive_path)
    with replay_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(observations):
        raise ValueError("replay CSV and synchronized observations differ in length")
    processor = LiveBEVProcessor()
    counts = Counter()
    maximum_distance_discrepancy_m = 0.0
    prior_timestamp_ns: Optional[int] = None
    examples = []
    with zipfile.ZipFile(archive_path) as archive:
        for index, (observation, exported) in enumerate(zip(observations, csv_rows)):
            if int(exported["frame_id"]) != observation.frame_id:
                raise ValueError("replay CSV frame order differs from sensor records")
            independent_speed_mps = observation.speed_kph / 3.6
            if not close(independent_speed_mps, observation.speed_mps):
                counts["gps_conversion_mismatches"] += 1
            exported_safety_speed_mps = csv_optional(
                exported, "safety_input_ego_speed_mps"
            )

            raw_dt_s = (
                0.05 if prior_timestamp_ns is None else
                (observation.monotonic_ns - prior_timestamp_ns) / 1e9
            )
            matrix = np.load(io.BytesIO(archive.read(observation.matrix_filename)))
            result = processor.update(
                matrix, dt=raw_dt_s, ego_speed_mps=observation.speed_mps
            )
            if (not close(observation.speed_mps, exported_safety_speed_mps)
                    or not close(
                        observation.speed_mps,
                        result.safety_result.input_ego_speed_mps,
                    )):
                counts["gps_to_safety_mismatches"] += 1
            manual_range_m = manual_bev_range_m(result.cleaned_matrix)
            adapter = extract_bev_longitudinal_observation(result.cleaned_matrix)
            adapter_range_m = adapter.bev_origin_obstacle_range_m
            core_gap_m = result.safety_result.input_obstacle_gap_m
            exported_adapter_range_m = csv_optional(
                exported, "bev_origin_obstacle_range_m"
            )
            exported_core_gap_m = csv_optional(
                exported, "safety_input_obstacle_gap_m"
            )
            if (not close(manual_range_m, adapter_range_m)
                    or not close(adapter_range_m, exported_adapter_range_m)):
                counts["manual_adapter_mismatches"] += 1
            if (not close(adapter.obstacle_gap_m, core_gap_m)
                    or not close(core_gap_m, exported_core_gap_m)):
                counts["adapter_core_mismatches"] += 1
            if manual_range_m is not None and adapter_range_m is not None:
                maximum_distance_discrepancy_m = max(
                    maximum_distance_discrepancy_m,
                    abs(manual_range_m - adapter_range_m),
                )

            csv_gap_m = exported_core_gap_m
            expected = oracle_evaluate(
                independent_speed_mps,
                csv_gap_m,
                float(exported["nominal_target_speed_mph"]),
            )
            if not close(
                expected.collision_boundary_m,
                float(exported["collision_boundary_m"]),
            ):
                counts["recorded_collision_mismatches"] += 1
            if not close(
                expected.critical_boundary_m,
                float(exported["critical_boundary_m"]),
            ):
                counts["recorded_critical_mismatches"] += 1
            if expected.zone != exported["zone"]:
                counts["recorded_zone_mismatches"] += 1
            if not close(
                expected.safe_target_speed_mph,
                float(exported["safe_target_speed_mph"]),
            ):
                counts["recorded_target_mismatches"] += 1
            examples.append({
                "frame": observation.frame_id,
                "speed_mph": independent_speed_mps / MPH_TO_MPS,
                "gap_m": csv_gap_m,
                "collision_m": expected.collision_boundary_m,
                "critical_m": expected.critical_boundary_m,
                "zone": expected.zone,
            })
            prior_timestamp_ns = observation.monotonic_ns
    counts["frames"] = len(observations)
    counts["maximum_distance_discrepancy_m"] = maximum_distance_discrepancy_m
    return dict(counts), select_speed_change_examples(examples)


def select_speed_change_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(
        range(1, len(rows)),
        key=lambda i: abs(float(rows[i]["speed_mph"]) - float(rows[i - 1]["speed_mph"])),
        reverse=True,
    )
    selected_indices: list[int] = []
    for index in ranked:
        for candidate in (index - 1, index):
            if candidate not in selected_indices:
                selected_indices.append(candidate)
        if len(selected_indices) >= 8:
            break
    return [rows[index] for index in sorted(selected_indices)]


def validate_uncertainty() -> tuple[int, int]:
    cases = (
        LongitudinalUncertainty(speed_uncertainty_mps=1.0),
        LongitudinalUncertainty(gap_uncertainty_m=1.0),
        LongitudinalUncertainty(delay_uncertainty_s=0.2),
    )
    comparisons = violations = 0
    for mph in range(41):
        for gap_m in range(81):
            state = LongitudinalSafetyState(
                mph * MPH_TO_MPS, float(gap_m), NOMINAL_TARGET_MPH, 80.0
            )
            baseline = evaluate_longitudinal_state(state)
            for uncertainty in cases:
                candidate = evaluate_longitudinal_state(state, uncertainty=uncertainty)
                comparisons += 1
                if (ZONE_RANK[candidate.zone] < ZONE_RANK[baseline.zone]
                        or candidate.safe_target_speed_mph
                        > baseline.safe_target_speed_mph + ABS_TOL):
                    violations += 1
    return comparisons, violations


def markdown_report(
    grid: dict[str, object],
    boundary_table: list[tuple],
    semantics_mismatches: int,
    monotonic: bool,
    sampled: list[tuple[int, str]],
    thresholds: list[tuple],
    recorded: dict[str, object],
    recorded_examples: list[dict[str, object]],
    uncertainty_comparisons: int,
    uncertainty_violations: int,
) -> str:
    lines = [
        "# Dynamic Zone Correctness Validation",
        "",
        "> This validation assesses correctness of the longitudinal safety algorithm with respect to its supplied state estimates. It does not establish accuracy of the upstream perception estimates.",
        "",
        "## Algorithm correctness",
        "",
        f"Independent tolerance: absolute `{ABS_TOL:g}`; boundary probe epsilon: `{BOUNDARY_EPSILON_M:g} m`.",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Synthetic states | {grid['states']} |",
        f"| Boundary mismatches | {grid['boundary_mismatches']} |",
        f"| Zone mismatches | {grid['zone_mismatches']} |",
        f"| Safe-target mismatches | {grid['target_mismatches']} |",
        f"| Maximum boundary error | {grid['maximum_boundary_error_m']:.3e} m |",
        f"| Maximum safe-target error | {grid['maximum_target_error_mph']:.3e} mph |",
        f"| Exact-boundary semantic mismatches (30 probes) | {semantics_mismatches} |",
        f"| Boundaries nondecreasing with sampled speed | {monotonic} |",
        "",
        "The oracle implements the accepted equations locally with basic `math`; it does not call production boundary/evaluation helpers for expected answers.",
        "",
        "### Dynamic boundaries",
        "",
        "| Speed | Collision boundary | Critical boundary |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {mph} mph | {collision:.6f} m | {critical:.6f} m |"
        for mph, collision, critical in boundary_table
    )
    lines.extend([
        "",
        "### Fixed-gap integer-speed sweep",
        "",
        "| Gap | Zones over sampled 0–40 mph |",
        "|---:|---|",
    ])
    lines.extend(f"| {gap} m | {zones} |" for gap, zones in sampled)
    lines.extend([
        "",
        "Analytic continuous transition thresholds independently solving each boundary equality:",
        "",
        "| Gap | SAFE below | CRITICAL range | COLLISION at/above |",
        "|---:|---:|---:|---:|",
    ])
    lines.extend(
        f"| {gap:.0f} m | {critical:.6f} mph | [{critical:.6f}, {collision:.6f}) mph | {collision:.6f} mph |"
        for gap, critical, collision in thresholds
    )
    lines.extend([
        "",
        "At the collision boundary equality the result is COLLISION; at critical equality it is CRITICAL; immediately beyond critical it is SAFE. Probes at 5, 10, 20, 30, and 40 mph produced no semantic mismatch.",
        "",
        "### Recorded interface and oracle checks",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Frames checked | {recorded['frames']} |",
        f"| GPS kph-to-m/s mismatches | {recorded.get('gps_conversion_mismatches', 0)} |",
        f"| GPS-to-safety-input mismatches | {recorded.get('gps_to_safety_mismatches', 0)} |",
        f"| Manual cleaned-BEV-to-adapter mismatches | {recorded.get('manual_adapter_mismatches', 0)} |",
        f"| Adapter-to-safety-core gap mismatches | {recorded.get('adapter_core_mismatches', 0)} |",
        f"| Maximum BEV distance discrepancy | {recorded['maximum_distance_discrepancy_m']:.3e} m |",
        f"| Recorded collision-boundary mismatches | {recorded.get('recorded_collision_mismatches', 0)} |",
        f"| Recorded critical-boundary mismatches | {recorded.get('recorded_critical_mismatches', 0)} |",
        f"| Recorded zone mismatches | {recorded.get('recorded_zone_mismatches', 0)} |",
        f"| Recorded safe-target mismatches | {recorded.get('recorded_target_mismatches', 0)} |",
        "",
        "Manual BEV extraction independently scans classes 1–6, requires `x_forward=80-row > 0`, and `|y_right=col-40| <= 2 m`. No self mask or physical-truth judgment is applied.",
        "",
        "### Recorded GPS dynamic examples",
        "",
        "The following frames come from the largest consecutive recorded speed changes; their boundaries are independently recalculated per frame.",
        "",
        "| Frame | GPS speed | Supplied BEV gap | Collision | Critical | Zone |",
        "|---:|---:|---:|---:|---:|---|",
    ])
    lines.extend(
        f"| {row['frame']} | {row['speed_mph']:.6f} mph | "
        f"{format_gap(row['gap_m'])} | "
        f"{row['collision_m']:.6f} m | {row['critical_m']:.6f} m | {row['zone']} |"
        for row in recorded_examples
    )
    lines.extend([
        "",
        "### Synthetic bounded-uncertainty consistency",
        "",
        f"Across {uncertainty_comparisons} comparisons (+1 m/s speed, +1 m gap, or +0.2 s delay uncertainty), less-conservative results: **{uncertainty_violations}**.",
        "",
        "## Upstream perception accuracy",
        "",
        "**NOT EVALUATED BY THIS TEST.** A supplied 2 m cleaned-BEV estimate is validated as the numeric 2 m safety input. This report does not determine whether the corresponding physical obstacle exists or is actually 2 m away.",
        "",
        "## Conclusion",
        "",
        "**ALGORITHM CORRECTNESS: " + ("PASS" if all([
            grid['boundary_mismatches'] == 0,
            grid['zone_mismatches'] == 0,
            grid['target_mismatches'] == 0,
            semantics_mismatches == 0,
            monotonic,
            recorded.get('gps_conversion_mismatches', 0) == 0,
            recorded.get('gps_to_safety_mismatches', 0) == 0,
            recorded.get('manual_adapter_mismatches', 0) == 0,
            recorded.get('adapter_core_mismatches', 0) == 0,
            recorded.get('recorded_collision_mismatches', 0) == 0,
            recorded.get('recorded_critical_mismatches', 0) == 0,
            recorded.get('recorded_zone_mismatches', 0) == 0,
            recorded.get('recorded_target_mismatches', 0) == 0,
            uncertainty_violations == 0,
        ]) else "FAIL") + "**",
        "",
        "**UPSTREAM PERCEPTION ACCURACY: NOT EVALUATED BY THIS TEST**",
        "",
    ])
    return "\n".join(lines)


def format_gap(gap_m: Optional[float]) -> str:
    return "none" if gap_m is None else f"{gap_m:.1f} m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="matrix_imu_gps.zip", type=Path)
    parser.add_argument(
        "--replay-csv", default="gps_longitudinal_safety_replay.csv", type=Path
    )
    parser.add_argument(
        "--grid-csv", default="dynamic_zone_validation.csv", type=Path
    )
    parser.add_argument(
        "--report", default="DYNAMIC_ZONE_CORRECTNESS_VALIDATION.md", type=Path
    )
    args = parser.parse_args()
    grid = validate_grid(args.grid_csv)
    boundary_table, semantics_mismatches, monotonic = validate_boundaries_and_semantics()
    sampled = fixed_gap_sampled_zones()
    thresholds = exact_gap_thresholds()
    recorded, examples = validate_recorded(args.archive, args.replay_csv)
    uncertainty_comparisons, uncertainty_violations = validate_uncertainty()
    report = markdown_report(
        grid, boundary_table, semantics_mismatches, monotonic, sampled,
        thresholds, recorded, examples, uncertainty_comparisons,
        uncertainty_violations,
    )
    args.report.write_text(report, encoding="utf-8")
    print(report)
    mismatch_total = (
        int(grid["boundary_mismatches"])
        + int(grid["zone_mismatches"])
        + int(grid["target_mismatches"])
        + semantics_mismatches
        + sum(int(recorded.get(key, 0)) for key in (
            "gps_conversion_mismatches", "gps_to_safety_mismatches",
            "manual_adapter_mismatches", "adapter_core_mismatches",
            "recorded_collision_mismatches", "recorded_critical_mismatches",
            "recorded_zone_mismatches", "recorded_target_mismatches",
        ))
        + uncertainty_violations
    )
    raise SystemExit(0 if mismatch_total == 0 and monotonic else 1)


if __name__ == "__main__":
    main()
