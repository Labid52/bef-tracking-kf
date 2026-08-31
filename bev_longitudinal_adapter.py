"""Thin BEV-to-physical-state adapter for longitudinal safety."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


ROWS, COLS = 120, 80
EGO_ROW, EGO_COL = 80, 40
CELL_SIZE_M = 1.0
NOMINAL_BEV_FORWARD_OBSERVATION_RANGE_M = EGO_ROW * CELL_SIZE_M
OBSTACLE_CLASSES = frozenset({1, 2, 3, 4, 5, 6})
CAR_CLASS_ID = 3


@dataclass(frozen=True)
class SelfVehicleExclusionConfig:
    x_forward_min_m: float
    x_forward_max_m: float
    lateral_half_width_m: float

    def validate(self) -> None:
        values = (self.x_forward_min_m, self.x_forward_max_m,
                  self.lateral_half_width_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("self-vehicle exclusion bounds must be finite")
        if self.x_forward_min_m < 0.0:
            raise ValueError("x_forward_min_m must be non-negative")
        if self.x_forward_max_m < self.x_forward_min_m:
            raise ValueError("self-vehicle exclusion longitudinal bounds are inconsistent")
        if self.lateral_half_width_m < 0.0:
            raise ValueError("lateral_half_width_m must be non-negative")


LEGACY_BROAD_CAR_EXCLUSION = SelfVehicleExclusionConfig(0.0, 6.0, 2.0)


@dataclass(frozen=True)
class BEVLongitudinalAdapterConfig:
    corridor_width_m: float = 4.0
    self_vehicle_exclusion: Optional[SelfVehicleExclusionConfig] = None
    # Unknown for the present roof/dashboard-area camera installation.  Do not
    # infer this from vehicle length or wheelbase.
    bev_origin_to_front_bumper_m: Optional[float] = None
    bev_forward_observation_range_m: float = NOMINAL_BEV_FORWARD_OBSERVATION_RANGE_M

    def validate(self) -> None:
        if not math.isfinite(self.corridor_width_m) or self.corridor_width_m <= 0.0:
            raise ValueError("corridor_width_m must be finite and positive")
        if self.self_vehicle_exclusion is not None:
            self.self_vehicle_exclusion.validate()
        if (not math.isfinite(self.bev_forward_observation_range_m)
                or self.bev_forward_observation_range_m < 0.0):
            raise ValueError(
                "bev_forward_observation_range_m must be finite and non-negative"
            )
        if self.bev_origin_to_front_bumper_m is not None:
            if (not math.isfinite(self.bev_origin_to_front_bumper_m)
                    or self.bev_origin_to_front_bumper_m < 0.0):
                raise ValueError(
                    "bev_origin_to_front_bumper_m must be finite and non-negative"
                )
            if self.bev_origin_to_front_bumper_m > self.bev_forward_observation_range_m:
                raise ValueError(
                    "bev_origin_to_front_bumper_m must not exceed forward observation range"
                )


@dataclass(frozen=True)
class BEVLongitudinalObservation:
    bev_origin_obstacle_range_m: Optional[float]
    obstacle_gap_m: Optional[float]
    bev_origin_observation_horizon_m: float
    observation_horizon_m: float
    front_bumper_offset_calibrated: bool
    gap_description: str
    horizon_description: str


def extract_bev_longitudinal_observation(
    matrix: np.ndarray,
    config: Optional[BEVLongitudinalAdapterConfig] = None,
) -> BEVLongitudinalObservation:
    """Extract nearest corridor obstacle and adapt it to a core gap input."""
    cfg = config or BEVLongitudinalAdapterConfig()
    cfg.validate()
    bev_range_m = nearest_bev_origin_obstacle_range_m(matrix, cfg)
    calibrated = cfg.bev_origin_to_front_bumper_m is not None
    bev_horizon_m = cfg.bev_forward_observation_range_m
    if bev_range_m is None:
        gap_m = None
    elif calibrated:
        gap_m = max(0.0, bev_range_m - cfg.bev_origin_to_front_bumper_m)
    else:
        # Compatibility behavior: retain the old numerical range, but do not
        # claim it is calibrated front-bumper clearance.
        gap_m = bev_range_m
    horizon_m = (
        max(0.0, bev_horizon_m - cfg.bev_origin_to_front_bumper_m)
        if calibrated else bev_horizon_m
    )
    return BEVLongitudinalObservation(
        bev_origin_obstacle_range_m=bev_range_m,
        obstacle_gap_m=gap_m,
        bev_origin_observation_horizon_m=bev_horizon_m,
        observation_horizon_m=horizon_m,
        front_bumper_offset_calibrated=calibrated,
        gap_description=(
            "front-bumper longitudinal gap"
            if calibrated else "provisional BEV-origin obstacle range"
        ),
        horizon_description=(
            "front-bumper observation horizon"
            if calibrated else "provisional BEV-origin observation horizon"
        ),
    )


def nearest_bev_origin_obstacle_range_m(
    matrix: np.ndarray,
    config: Optional[BEVLongitudinalAdapterConfig] = None,
) -> Optional[float]:
    cfg = config or BEVLongitudinalAdapterConfig()
    cfg.validate()
    m = np.asarray(matrix)
    if m.shape != (ROWS, COLS):
        raise ValueError(f"matrix must have shape {(ROWS, COLS)}, got {m.shape}")
    if m.dtype != np.uint8:
        raise TypeError(f"matrix must have dtype uint8, got {m.dtype}")
    rows, cols = np.where(np.isin(m, tuple(OBSTACLE_CLASSES)))
    if rows.size == 0:
        return None
    classes = m[rows, cols]
    forward_m = (EGO_ROW - rows).astype(float) * CELL_SIZE_M
    lateral_m = (cols - EGO_COL).astype(float) * CELL_SIZE_M
    valid = (forward_m > 0.0) & (np.abs(lateral_m) <= cfg.corridor_width_m / 2.0)
    if cfg.self_vehicle_exclusion is not None:
        mask = cfg.self_vehicle_exclusion
        artifact = (
            (classes == CAR_CLASS_ID)
            & (forward_m >= mask.x_forward_min_m)
            & (forward_m <= mask.x_forward_max_m)
            & (np.abs(lateral_m) <= mask.lateral_half_width_m)
        )
        valid &= ~artifact
    return float(np.min(forward_m[valid])) if np.any(valid) else None
