"""Compatibility entry point: cleaned BEV adapter -> generic safety core.

New non-BEV callers should import :func:`evaluate_longitudinal_state` from
``longitudinal_safety_core``. This module preserves the original matrix API.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from bev_longitudinal_adapter import (
    BEVLongitudinalAdapterConfig,
    CAR_CLASS_ID,
    CELL_SIZE_M,
    COLS,
    EGO_COL,
    EGO_ROW,
    LEGACY_BROAD_CAR_EXCLUSION,
    NOMINAL_BEV_FORWARD_OBSERVATION_RANGE_M,
    OBSTACLE_CLASSES,
    ROWS,
    SelfVehicleExclusionConfig,
    extract_bev_longitudinal_observation,
    nearest_bev_origin_obstacle_range_m,
)
from longitudinal_safety_core import (
    COLLISION,
    CRITICAL,
    HORIZON_CRITICAL_LIMITED,
    HORIZON_EMERGENCY_LIMITED,
    HORIZON_FULL,
    SAFE,
    LEGACY_PROVISIONAL_V1_CONFIG,
    LongitudinalSafetyConfig,
    LongitudinalSafetyResult,
    LongitudinalSafetyState,
    LongitudinalUncertainty,
    VehicleLongitudinalSpec,
    dynamic_boundaries_m,
    evaluate_longitudinal_state,
)


MPS_TO_MPH = 1.0 / 0.44704


def nearest_longitudinal_obstacle_m(
    matrix: np.ndarray,
    config: Optional[BEVLongitudinalAdapterConfig] = None,
) -> Optional[float]:
    """Compatibility alias for the uncalibrated BEV-origin obstacle range."""
    return nearest_bev_origin_obstacle_range_m(matrix, config)


def evaluate_longitudinal_safety(
    matrix: np.ndarray,
    ego_speed_mps: float,
    nominal_target_speed_mph: float,
    dt: Optional[float] = None,
    config: Optional[LongitudinalSafetyConfig] = None,
    uncertainty: Optional[LongitudinalUncertainty] = None,
    bev_config: Optional[BEVLongitudinalAdapterConfig] = None,
) -> LongitudinalSafetyResult:
    """Preserved matrix API implemented as adapter then generic-core call.

    Without a calibrated ``bev_origin_to_front_bumper_m``, the adapter passes
    the BEV-origin obstacle range as a provisional gap for replay compatibility.
    """
    if dt is not None and (not math.isfinite(float(dt)) or float(dt) <= 0.0):
        raise ValueError("dt must be finite and positive when supplied")
    observation = extract_bev_longitudinal_observation(matrix, bev_config)
    return evaluate_longitudinal_state(
        LongitudinalSafetyState(
            ego_speed_mps=ego_speed_mps,
            obstacle_gap_m=observation.obstacle_gap_m,
            nominal_target_speed_mph=nominal_target_speed_mph,
            observation_horizon_m=observation.observation_horizon_m,
        ),
        config=config,
        uncertainty=uncertainty,
    )
