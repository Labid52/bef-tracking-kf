"""BEV-independent longitudinal three-zone safety mathematics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


SAFE = "SAFE"
CRITICAL = "CRITICAL"
COLLISION = "COLLISION"
HORIZON_FULL = "FULL"
HORIZON_CRITICAL_LIMITED = "CRITICAL_LIMITED"
HORIZON_EMERGENCY_LIMITED = "EMERGENCY_LIMITED"
_RANGE_CONSISTENCY_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class VehicleLongitudinalSpec:
    """Vehicle-team values; geometry is stored but not used by scalar V1."""

    vehicle_width_m: float = 1.89
    vehicle_length_m: float = 4.66
    wheelbase_m: float = 3.00
    comfortable_deceleration_mps2: float = 2.0
    emergency_deceleration_mps2: float = 9.0

    def validate(self) -> None:
        values = {
            "vehicle_width_m": self.vehicle_width_m,
            "vehicle_length_m": self.vehicle_length_m,
            "wheelbase_m": self.wheelbase_m,
            "comfortable_deceleration_mps2": self.comfortable_deceleration_mps2,
            "emergency_deceleration_mps2": self.emergency_deceleration_mps2,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.wheelbase_m > self.vehicle_length_m:
            raise ValueError("wheelbase_m must not exceed vehicle_length_m")
        if self.comfortable_deceleration_mps2 > self.emergency_deceleration_mps2:
            raise ValueError(
                "comfortable_deceleration_mps2 must not exceed "
                "emergency_deceleration_mps2"
            )


@dataclass(frozen=True)
class LongitudinalSafetyConfig:
    """Provisional deterministic parameters for the generic safety core."""

    vehicle_spec: VehicleLongitudinalSpec = field(default_factory=VehicleLongitudinalSpec)
    standstill_clearance_m: float = 2.0
    machine_response_delay_s: float = 0.30
    critical_extra_margin_s: float = 0.45

    def validate(self) -> None:
        self.vehicle_spec.validate()
        for name, value in {
            "standstill_clearance_m": self.standstill_clearance_m,
            "machine_response_delay_s": self.machine_response_delay_s,
            "critical_extra_margin_s": self.critical_extra_margin_s,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def critical_total_intervention_time_s(self) -> float:
        return self.machine_response_delay_s + self.critical_extra_margin_s


LEGACY_PROVISIONAL_V1_CONFIG = LongitudinalSafetyConfig(
    vehicle_spec=VehicleLongitudinalSpec(
        comfortable_deceleration_mps2=3.0,
        emergency_deceleration_mps2=6.0,
    ),
    standstill_clearance_m=2.0,
    machine_response_delay_s=0.30,
    critical_extra_margin_s=0.45,
)


@dataclass(frozen=True)
class LongitudinalSafetyState:
    """Physical scalar inputs plus an optional finite observation horizon.

    A ``None`` gap means no relevant obstacle was supplied or observed; the
    horizon distinguishes the unbounded synthetic and finite-observation cases.
    """

    ego_speed_mps: float
    obstacle_gap_m: Optional[float]
    nominal_target_speed_mph: float
    # None explicitly means an unbounded synthetic observation assumption.
    observation_horizon_m: Optional[float] = None


@dataclass(frozen=True)
class LongitudinalUncertainty:
    """Non-negative deterministic uncertainty bounds."""

    speed_uncertainty_mps: float = 0.0
    gap_uncertainty_m: float = 0.0
    delay_uncertainty_s: float = 0.0

    def validate(self) -> None:
        for name, value in {
            "speed_uncertainty_mps": self.speed_uncertainty_mps,
            "gap_uncertainty_m": self.gap_uncertainty_m,
            "delay_uncertainty_s": self.delay_uncertainty_s,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class LongitudinalSafetyResult:
    zone: str
    input_ego_speed_mps: float
    effective_ego_speed_mps: float
    input_obstacle_gap_m: Optional[float]
    effective_obstacle_gap_m: Optional[float]
    observation_horizon_m: Optional[float]
    collision_boundary_m: float
    critical_boundary_m: float
    nominal_target_speed_mph: float
    safe_target_speed_mph: float
    machine_response_delay_s: float
    effective_machine_response_delay_s: float
    maximum_allowable_delay_s: Optional[float]
    delay_margin_s: Optional[float]
    observable_horizon_maximum_delay_s: Optional[float]
    observable_horizon_delay_margin_s: Optional[float]
    collision_horizon_covered: bool
    critical_horizon_covered: bool
    coverage_status: str
    maximum_collision_horizon_speed_mps: Optional[float]
    maximum_critical_horizon_speed_mps: Optional[float]
    collision_horizon_speed_margin_mps: Optional[float]
    critical_horizon_speed_margin_mps: Optional[float]
    reason: str

    # Compatibility aliases used by the existing live result/viewer.
    @property
    def ego_speed_mps(self) -> float:
        return self.input_ego_speed_mps

    @property
    def obstacle_distance_m(self) -> Optional[float]:
        return self.input_obstacle_gap_m


def dynamic_boundaries_m(
    ego_speed_mps: float,
    config: Optional[LongitudinalSafetyConfig] = None,
    *,
    effective_machine_response_delay_s: Optional[float] = None,
) -> tuple[float, float]:
    """Return collision and critical boundaries for physical scalar inputs."""
    cfg = config or LongitudinalSafetyConfig()
    cfg.validate()
    speed_mps = _finite_nonnegative("ego_speed_mps", ego_speed_mps)
    machine_delay_s = (
        cfg.machine_response_delay_s
        if effective_machine_response_delay_s is None
        else _finite_nonnegative(
            "effective_machine_response_delay_s", effective_machine_response_delay_s
        )
    )
    spec = cfg.vehicle_spec
    collision_m = (
        cfg.standstill_clearance_m
        + speed_mps * machine_delay_s
        + speed_mps**2 / (2.0 * spec.emergency_deceleration_mps2)
    )
    critical_m = (
        cfg.standstill_clearance_m
        + speed_mps * (machine_delay_s + cfg.critical_extra_margin_s)
        + speed_mps**2 / (2.0 * spec.comfortable_deceleration_mps2)
    )
    if critical_m + 1e-12 < collision_m:
        raise ValueError("configuration produces critical boundary below collision boundary")
    return float(collision_m), float(critical_m)


def evaluate_longitudinal_state(
    state: LongitudinalSafetyState,
    config: Optional[LongitudinalSafetyConfig] = None,
    uncertainty: Optional[LongitudinalUncertainty] = None,
) -> LongitudinalSafetyResult:
    """Evaluate the generic safety model without perception or BEV dependencies."""
    cfg = config or LongitudinalSafetyConfig()
    unc = uncertainty or LongitudinalUncertainty()
    cfg.validate()
    unc.validate()
    input_speed_mps = _finite_nonnegative("ego_speed_mps", state.ego_speed_mps)
    nominal_mph = _finite_nonnegative(
        "nominal_target_speed_mph", state.nominal_target_speed_mph
    )
    input_gap_m = state.obstacle_gap_m
    if input_gap_m is not None:
        input_gap_m = _finite_nonnegative("obstacle_gap_m", input_gap_m)
    observation_horizon_m = state.observation_horizon_m
    if observation_horizon_m is not None:
        observation_horizon_m = _finite_nonnegative(
            "observation_horizon_m", observation_horizon_m
        )
        if (input_gap_m is not None
                and input_gap_m > observation_horizon_m + _RANGE_CONSISTENCY_TOLERANCE_M):
            raise ValueError(
                "obstacle_gap_m must not exceed finite observation_horizon_m"
            )

    effective_speed_mps = input_speed_mps + unc.speed_uncertainty_mps
    effective_gap_m = (
        None if input_gap_m is None
        else max(0.0, input_gap_m - unc.gap_uncertainty_m)
    )
    effective_delay_s = cfg.machine_response_delay_s + unc.delay_uncertainty_s
    collision_m, critical_m = dynamic_boundaries_m(
        effective_speed_mps,
        cfg,
        effective_machine_response_delay_s=effective_delay_s,
    )

    if effective_gap_m is None and observation_horizon_m is None:
        zone, safe_mph = SAFE, nominal_mph
        reason = "no obstacle supplied under explicit unbounded synthetic observation"
    elif effective_gap_m is None:
        zone, safe_mph = SAFE, nominal_mph
        reason = "no relevant obstacle observed within forward observation horizon"
    elif effective_gap_m <= collision_m:
        zone, safe_mph = COLLISION, 0.0
        reason = "effective gap at or inside collision boundary"
    elif effective_gap_m <= critical_m:
        zone = CRITICAL
        fraction = (effective_gap_m - collision_m) / (critical_m - collision_m)
        safe_mph = nominal_mph * math.sqrt(min(max(fraction, 0.0), 1.0))
        reason = "effective gap inside critical boundary; speed cap applied"
    else:
        zone, safe_mph = SAFE, nominal_mph
        reason = "effective gap beyond critical boundary"

    if effective_gap_m is None and observation_horizon_m is not None:
        maximum_delay_s, delay_margin_s = None, None
    else:
        maximum_delay_s, delay_margin_s = _delay_diagnostics(
            effective_speed_mps,
            effective_gap_m,
            effective_delay_s,
            cfg,
        )
    if observation_horizon_m is None:
        observable_delay_s, observable_delay_margin_s = None, None
        collision_covered = critical_covered = True
        coverage_status = HORIZON_FULL
        max_collision_speed_mps = max_critical_speed_mps = math.inf
    else:
        observable_delay_s, observable_delay_margin_s = _delay_diagnostics(
            effective_speed_mps,
            observation_horizon_m,
            effective_delay_s,
            cfg,
        )
        collision_covered = observation_horizon_m >= collision_m
        critical_covered = observation_horizon_m >= critical_m
        coverage_status = (
            HORIZON_FULL if critical_covered
            else HORIZON_CRITICAL_LIMITED if collision_covered
            else HORIZON_EMERGENCY_LIMITED
        )
        max_collision_speed_mps = maximum_horizon_input_speed_mps(
            observation_horizon_m,
            cfg.standstill_clearance_m,
            effective_delay_s,
            cfg.vehicle_spec.emergency_deceleration_mps2,
            unc.speed_uncertainty_mps,
        )
        max_critical_speed_mps = maximum_horizon_input_speed_mps(
            observation_horizon_m,
            cfg.standstill_clearance_m,
            effective_delay_s + cfg.critical_extra_margin_s,
            cfg.vehicle_spec.comfortable_deceleration_mps2,
            unc.speed_uncertainty_mps,
        )
    collision_speed_margin_mps = (
        None if max_collision_speed_mps is None
        else max_collision_speed_mps - input_speed_mps
    )
    critical_speed_margin_mps = (
        None if max_critical_speed_mps is None
        else max_critical_speed_mps - input_speed_mps
    )
    safe_mph = min(nominal_mph, max(0.0, float(safe_mph)))
    return LongitudinalSafetyResult(
        zone=zone,
        input_ego_speed_mps=input_speed_mps,
        effective_ego_speed_mps=effective_speed_mps,
        input_obstacle_gap_m=input_gap_m,
        effective_obstacle_gap_m=effective_gap_m,
        observation_horizon_m=observation_horizon_m,
        collision_boundary_m=collision_m,
        critical_boundary_m=critical_m,
        nominal_target_speed_mph=nominal_mph,
        safe_target_speed_mph=safe_mph,
        machine_response_delay_s=cfg.machine_response_delay_s,
        effective_machine_response_delay_s=effective_delay_s,
        maximum_allowable_delay_s=maximum_delay_s,
        delay_margin_s=delay_margin_s,
        observable_horizon_maximum_delay_s=observable_delay_s,
        observable_horizon_delay_margin_s=observable_delay_margin_s,
        collision_horizon_covered=collision_covered,
        critical_horizon_covered=critical_covered,
        coverage_status=coverage_status,
        maximum_collision_horizon_speed_mps=max_collision_speed_mps,
        maximum_critical_horizon_speed_mps=max_critical_speed_mps,
        collision_horizon_speed_margin_mps=collision_speed_margin_mps,
        critical_horizon_speed_margin_mps=critical_speed_margin_mps,
        reason=reason,
    )


def maximum_horizon_input_speed_mps(
    observation_horizon_m: float,
    standstill_clearance_m: float,
    total_delay_s: float,
    deceleration_mps2: float,
    speed_uncertainty_mps: float = 0.0,
) -> Optional[float]:
    """Maximum input speed whose dynamic boundary fits inside the horizon.

    The quadratic is solved for maximum *effective* speed, then the bounded
    speed uncertainty is subtracted. ``None`` means even zero input speed is
    incompatible because clearance plus the uncertainty-induced boundary is
    already beyond the horizon.
    """
    horizon_m = _finite_nonnegative("observation_horizon_m", observation_horizon_m)
    clearance_m = _finite_nonnegative("standstill_clearance_m", standstill_clearance_m)
    delay_s = _finite_nonnegative("total_delay_s", total_delay_s)
    deceleration = float(deceleration_mps2)
    if not math.isfinite(deceleration) or deceleration <= 0.0:
        raise ValueError("deceleration_mps2 must be finite and positive")
    uncertainty_mps = _finite_nonnegative(
        "speed_uncertainty_mps", speed_uncertainty_mps
    )
    available_m = horizon_m - clearance_m
    if available_m < 0.0:
        return None
    maximum_effective_speed_mps = deceleration * (
        math.sqrt(delay_s**2 + 2.0 * available_m / deceleration) - delay_s
    )
    maximum_input_speed_mps = maximum_effective_speed_mps - uncertainty_mps
    if maximum_input_speed_mps < 0.0:
        return None
    return float(maximum_input_speed_mps)


def _delay_diagnostics(
    effective_speed_mps: float,
    effective_gap_m: Optional[float],
    effective_delay_s: float,
    config: LongitudinalSafetyConfig,
) -> tuple[Optional[float], Optional[float]]:
    if effective_speed_mps == 0.0:
        return None, None
    if effective_gap_m is None:
        return math.inf, math.inf
    emergency_braking_m = (
        effective_speed_mps**2
        / (2.0 * config.vehicle_spec.emergency_deceleration_mps2)
    )
    maximum_delay_s = (
        effective_gap_m - config.standstill_clearance_m - emergency_braking_m
    ) / effective_speed_mps
    return float(maximum_delay_s), float(maximum_delay_s - effective_delay_s)


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value
