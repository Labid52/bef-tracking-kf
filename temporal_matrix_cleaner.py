#!/usr/bin/env python3
"""
Temporal cleaning / tracking for 120x80 ego-relative semantic BEV matrices.

The input rasters are sparse point detections (measured: >99.8% of connected
components are a single cell), sampled at 10 FPS from a front-camera
perception stack whose effective update rate is ~5 Hz.  This module recovers
persistent object tracks with continuous sub-cell state, then re-rasterises
them onto the ORIGINAL 120x80 / 1 m grid so that the unmodified
``target_speed.getTargetSpeed`` keeps its exact physical meaning.

Pipeline
--------
  0. detections      : nonzero cells -> (x_forward, y_right, class)
  1. bootstrap pass  : ego-frame constant-velocity tracker (no ego motion)
  2. ego motion      : robust (v, yaw-rate) per frame from the bootstrap
                       velocity field, then smoothed
  3. main pass       : Kalman tracker whose state is the object's WORLD
                       velocity expressed in the current ego frame, driven by
                       the estimated ego motion
  4. offline refine  : tracklet stitching, duplicate merging, static-object
                       constraint, RTS smoothing, retro-active track start
  5. rasterise       : back to 120x80 uint8, original convention

Coordinate convention (identical to target_speed.py, deliberately)
------------------------------------------------------------------
    x_forward_m = (EGO_ROW - row) * CELL_SIZE_M      row 80 -> 0 m
    y_right_m   = (col - EGO_COL) * CELL_SIZE_M      col 40 -> 0 m
    row = EGO_ROW - round(x)   ;   col = EGO_COL + round(y)
The forward/inverse maps are exact inverses on the integer grid, so a track
that never moves re-rasterises to the very cell it came from.

Motion model (x forward, y right, yaw-rate psi positive = turning right)
------------------------------------------------------------------------
Only the ROTATIONAL part of ego motion enters the model.  Over one step the
ego rotates by D = psi*dt, so for a point whose velocity relative to a
translating-but-non-rotating ego frame is w:

    p_{k+1} = M(D) @ (p_k + w_k*dt)
    w_{k+1} = M(D) @ w_k                 M(D) = [[cosD, sinD], [-sinD, cosD]]

Ego SPEED is deliberately NOT used as a model input.  It cannot be recovered
reliably here (vehicles moving with the ego contaminate the static set) and,
worse, imposing a wrong speed would shift object ranges and therefore silently
change the effective stopping distances getTargetSpeed was tuned for.  Each
track's own w absorbs the translation instead: a stationary world object simply
learns w = (-v_ego, 0).

That gives the stop-sign prior its clean form.  In the yaw-compensated frame a
STATIONARY world object has w_y = 0 exactly, whatever the ego speed is.  So
infrastructure classes get a lateral velocity that decays toward zero
(Ornstein-Uhlenbeck, correlation time tau_lateral) while their range motion is
left entirely to the data.  This is the formal version of "a sign must move
like a stationary object seen from a moving car", and it needs no odometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Geometry -- must stay identical to target_speed.py
# ---------------------------------------------------------------------------

#: Provenance of an emitted object state.  Diagnostics only -- it never
#: changes what is written into the control matrix, and the presentation
#: animation must not change an object's class colour because of it.
PROV_OBSERVED = "observed"          # a raw detection was assigned this frame
PROV_INTERPOLATED = "interpolated"  # offline: inside the observed span, filled
                                    # by the RTS smoother using past AND future
PROV_COASTED = "coasted"            # forward prediction past the last detection
PROV_PASSTHROUGH = "passthrough"    # restored verbatim by the safety pass
PROVENANCES = (PROV_OBSERVED, PROV_INTERPOLATED, PROV_COASTED, PROV_PASSTHROUGH)

ROWS, COLS = 120, 80
EGO_ROW, EGO_COL = 80, 40
CELL_SIZE_M = 1.0
DT = 0.10  # seconds per frame (10 FPS)

X_MAX = float(EGO_ROW)                      # +80 m
X_MIN = float(EGO_ROW - (ROWS - 1))         # -39 m
Y_MIN = float(-EGO_COL)                     # -40 m
Y_MAX = float(COLS - 1 - EGO_COL)           # +39 m

EMPTY = 0

# Class groups.  Association is allowed only inside a group; this is what
# absorbs the measured car<->truck label flicker without letting a pedestrian
# be swallowed by a car track.
GROUP_VRU = "vru"
GROUP_VEHICLE = "vehicle"
GROUP_SIGN = "sign"
GROUP_LIGHT = "light"
GROUP_OTHER = "other"

CLASS_GROUP: Dict[int, str] = {
    1: GROUP_VRU, 2: GROUP_VRU,
    3: GROUP_VEHICLE, 4: GROUP_VEHICLE, 5: GROUP_VEHICLE, 6: GROUP_VEHICLE,
    7: GROUP_SIGN,
    8: GROUP_LIGHT, 9: GROUP_LIGHT, 10: GROUP_LIGHT, 11: GROUP_LIGHT,
}

def class_group(cls: int) -> str:
    """Group of a semantic class; unknown/unlisted classes get their own group."""
    return CLASS_GROUP.get(int(cls), f"{GROUP_OTHER}{int(cls)}")

# Traffic-light classes genuinely change state over time, so their label must
# follow the most recent observation instead of a majority vote.
STATEFUL_GROUPS = frozenset({GROUP_LIGHT})

# Classes that getTargetSpeed reacts to (mirrored here for the safety zone
# only -- the policy itself lives in target_speed.py and is not redefined).
TS_OBSTACLE_CLASSES = frozenset({1, 2, 3, 4, 5, 6})
TS_STOP_CLASSES = frozenset({7, 9})

# Cell-conflict priority when two tracks land in the same cell.  Most
# conservative (largest braking demand) wins, so a conflict can never relax
# the target speed:  VRU > non-car vehicle > car > stop/red > light.
_CONFLICT_PRIORITY: Dict[int, int] = {
    1: 100, 2: 100,
    5: 90, 6: 90, 4: 90, 3: 80,
    7: 60, 9: 60,
    8: 20, 10: 20, 11: 20,
}

def _priority(cls: int) -> int:
    return _CONFLICT_PRIORITY.get(int(cls), 10)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CleanerConfig:
    """All tunables.  Defaults were derived from measurements on this dataset."""

    # --- measurement noise (polar: camera bearing is good, range is not) ---
    sigma_range_m: float = 1.20        # measured 1.0, inflated for the 5 Hz hold lag
    sigma_range_slope: float = 0.10    # extra range sigma per metre of range.
                                       # Monocular depth error grows with range:
                                       # measured, a static stop sign appears to
                                       # close at ~10 m/s inside 20 m but ~15 m/s
                                       # at 20-45 m, which is a pure range bias.
    sigma_tangential_m: float = 0.75   # measured 0.6
    quantisation_var: float = 1.0 / 12.0   # uniform 1 m cell
    robust_chi2: float = 9.21          # above this, downweight (M-estimator)

    # --- process noise: world-frame acceleration per group ---
    sigma_accel: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 3.0, GROUP_VRU: 1.5, GROUP_SIGN: 0.25,
        GROUP_LIGHT: 0.25, GROUP_OTHER: 2.0,
    })
    sigma_v0: float = 8.0              # initial world-velocity uncertainty (m/s)

    # --- association ---
    gate_chi2: float = 16.0            # 2 dof, ~99.97%
    # The absolute association gate is anisotropic, like everything else in this
    # module: range is the axis the sensor gets wrong, bearing is the axis it
    # gets right.  A gate that is generous in BOTH directions lets a long coast
    # swallow an unrelated detection; one that is tight in both breaks the
    # legitimate multi-metre range jumps this sensor produces every frame.
    gate_max_m: float = 6.0                  # radial gate at zero coast/range
    gate_range_frac: float = 0.15            # radial growth per metre of range
    gate_grow_m_per_frame: float = 1.2       # radial growth per coasted frame
    gate_tangential_m: float = 3.5           # cross-range gate at zero coast
    gate_tangential_range_frac: float = 0.06
    gate_tangential_per_frame: float = 0.5
    class_switch_penalty: float = 2.0  # cost added for a within-group label change

    # --- intra-frame duplicate clustering (conservative: measured only 84 of
    #     104107 same-group vehicle pairs are within 1.5 m) ---
    cluster_radius_m: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 2.0, GROUP_VRU: 2.0, GROUP_SIGN: 2.0,
        GROUP_LIGHT: 2.0, GROUP_OTHER: 2.0,
    })

    # --- track lifecycle ---
    confirm_hits: int = 3              # M of N
    confirm_window: int = 6
    max_coast_frames: Dict[str, int] = field(default_factory=lambda: {
        GROUP_VEHICLE: 12, GROUP_VRU: 10, GROUP_SIGN: 15,
        GROUP_LIGHT: 15, GROUP_OTHER: 10,
    })
    emit_coast_frames: int = 8         # CAUSAL only: trailing coast still drawn.
                                       # Offline emits nothing after the last
                                       # observation: with the future available
                                       # a gap that never closes is a genuine
                                       # disappearance, not a dropout.
    max_pos_sigma_m: float = 6.0       # kill a track once it is this uncertain
    grid_margin_m: float = 3.0         # kill a track this far outside the raster

    # --- safety zone: the region getTargetSpeed can actually react to.
    #     Inside it, an unconfirmed detection is NEVER suppressed. ---
    safety_max_forward_m: float = 50.0
    safety_obstacle_half_width_m: float = 4.0   # corridor is +-2 m; +2 m margin
    safety_stop_half_width_m: float = 17.0      # stop gate is +-15 m; +2 m margin

    # --- offline refinement ---
    # Stitch gates are anisotropic for the same reason the measurement noise is:
    # range is the untrustworthy axis, bearing is not.  A generous radial gate
    # lets one physical sign stay one track across the sensor's range jumps; a
    # tight tangential gate stops two objects at different bearings from being
    # glued together.
    stitch_max_gap_frames: int = 15
    stitch_gate_radial_m: float = 5.0
    stitch_gate_radial_frac: float = 0.15       # per metre of range
    stitch_gate_tangential_m: float = 3.5
    stitch_gate_per_frame_m: float = 0.4
    stitch_min_obs: int = 2            # never stitch onto a single detection
    merge_min_overlap_frames: int = 3
    max_fit_rms_m: float = 2.6         # a merge or a stitch is accepted only if
                                       # the combined track still explains its
                                       # own measurements this well.  Proximity
                                       # alone is not evidence that two
                                       # detections are one object.
    fit_degradation_factor: float = 1.5
    merge_radius_m: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 3.0, GROUP_VRU: 3.0, GROUP_SIGN: 6.0,
        GROUP_LIGHT: 6.0, GROUP_OTHER: 3.0,
    })
    # Duplicate suppression.  Merging asks "are these one object?"; suppression
    # asks the weaker question "is this track redundant given a better-supported
    # one?", which is what removes the split range hypotheses that produce
    # STOP / STOP / STOP.  A duplicate is only ever dropped in favour of a
    # clearly better-supported track of the same class group.
    duplicate_radius_m: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 3.0, GROUP_VRU: 3.0, GROUP_SIGN: 12.0,
        GROUP_LIGHT: 12.0, GROUP_OTHER: 3.0,
    })
    duplicate_min_overlap_frames: int = 4
    # Coverage radius for the safety pass-through.  Deliberately TIGHTER than
    # duplicate_radius_m: sustained co-location over many frames is evidence of
    # duplication, a single-frame offset is not, so a lone far detection must
    # not be allowed to "cover" (and thereby delete) a near one.
    passthrough_cover_radius_m: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 3.0, GROUP_VRU: 3.0, GROUP_SIGN: 6.0,
        GROUP_LIGHT: 6.0, GROUP_OTHER: 3.0,
    })
    duplicate_min_support: int = 8     # survivor needs this many observations ...
    duplicate_support_ratio: float = 2.0   # ... or this many times the loser's

    static_speed_mps: float = 1.5      # |w - (-v_ego, 0)| below this -> reported
                                       # as static (annotation only; it does not
                                       # drive the filter)
    static_prior_groups: Tuple[str, ...] = (GROUP_SIGN, GROUP_LIGHT)
    # Signs and traffic lights ARE stationary infrastructure.  Enforcing w = 0
    # as a class prior (instead of inferring it) is what stops a run of bad
    # range estimates from being explained as a fast-moving "sign".
    # Lateral-velocity correlation time for stationary infrastructure.  In the
    # yaw-compensated frame a static world object has w_y = 0 exactly, so w_y is
    # damped toward zero with this time constant.  w_x is left free, which keeps
    # the tracker faithful to the measured range (the sensor's range bias is not
    # something this stage is allowed to "correct" -- see the module docstring).
    tau_lateral_s: float = 0.30
    min_track_observations: int = 3    # below this a track is clutter (outside
                                       # the safety zone only)

    # Physical-plausibility caps on the recovered velocity.  These are an
    # ABSURDITY filter, not a tight prior: a real sign can show ~8 m/s of
    # apparent lateral rate here, because the ego turns and because the sensor's
    # range bias drags the projected lateral position with it.  Tighter caps
    # were measured to delete genuine, well-observed tracks.  Clutter is removed
    # by min_track_observations, the lateral damping and duplicate suppression
    # instead.  The radial cap is looser still: apparent radial speed is
    # inflated by the range bias, which this stage must not correct away.
    max_lateral_world_speed_mps: Dict[str, float] = field(default_factory=lambda: {
        GROUP_VEHICLE: 30.0, GROUP_VRU: 15.0, GROUP_SIGN: 10.0,
        GROUP_LIGHT: 10.0, GROUP_OTHER: 15.0,
    })
    max_radial_world_speed_mps: float = 60.0

    # --- ego motion ---
    ego_ransac_iters: int = 80
    ego_inlier_tol_mps: float = 2.5   # velocity residual tolerance for the static set
    ego_min_inliers: int = 3
    ego_sigma_v_meas: float = 2.5
    ego_sigma_a: float = 1.5
    ego_sigma_psi_meas: float = 0.08
    # A road vehicle's yaw ACCELERATION is bounded; 0.15 rad/s^2 lets the yaw
    # rate change by 0.15 rad/s per second, which covers normal steering while
    # rejecting the per-frame spikes the finite-difference flow can produce.
    ego_sigma_psidot: float = 0.15
    ego_causal_rate_damping: float = 0.0  # see ScalarCVFilter.__init__
    # Baseline (in frames) for the finite-difference velocity fed to the
    # ego-motion solver.  2 frames spans one full 5 Hz perception update, so the
    # measured hold-and-jump averages out instead of aliasing into the estimate.
    ego_flow_finite_difference: bool = True   # False reproduces the old
                                       # filtered-velocity behaviour (A/B only)
    ego_flow_baseline_frames: int = 2
    ego_flow_max_baseline_frames: int = 4
    ego_near_range_m: float = 35.0     # only points this close are used for v,
                                       # because the range bias grows with range
    ego_v_max: float = 30.0
    ego_psi_max: float = 0.8           # 46 deg/s: a physical bound at these speeds
    use_ego_motion: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x: float           # forward m
    y: float           # right m (positive = right)
    cls: int
    group: str
    n_cells: int = 1   # how many raw cells were merged into this measurement
    raw_cells: Tuple[Tuple[int, int, int], ...] = ()   # (row, col, class)


def extract_detections(matrix: np.ndarray, cfg: CleanerConfig) -> List[Detection]:
    """Nonzero cells -> point measurements, with a conservative intra-frame merge.

    Merging uses single-linkage inside a class group with a small radius.  It
    only removes the unambiguous case (a detection split across two adjacent
    cells); genuinely distinct objects of the same group are >= 3.5 m apart in
    this dataset in all but 565 of 104107 pairs.
    """
    m = np.asarray(matrix)
    rr, cc = np.nonzero(m)
    if rr.size == 0:
        return []

    raw = []
    for r, c in zip(rr.tolist(), cc.tolist()):
        cls = int(m[r, c])
        raw.append((float(EGO_ROW - r), float(c - EGO_COL), cls, r, c))
    # Deterministic order: forward-most first, then left-to-right, then class.
    raw.sort(key=lambda t: (-t[0], t[1], t[2]))

    n = len(raw)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        gi = class_group(raw[i][2])
        ri = cfg.cluster_radius_m.get(gi, 2.0)
        for j in range(i + 1, n):
            if class_group(raw[j][2]) != gi:
                continue
            d = math.hypot(raw[i][0] - raw[j][0], raw[i][1] - raw[j][1])
            if d <= ri:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    dets: List[Detection] = []
    for root in sorted(clusters):
        members = clusters[root]
        xs = np.array([raw[i][0] for i in members])
        ys = np.array([raw[i][1] for i in members])
        # Representative class: the one closest to ego wins (best-observed).
        rep = min(members, key=lambda i: (math.hypot(raw[i][0], raw[i][1]), i))
        dets.append(Detection(
            x=float(xs.mean()), y=float(ys.mean()),
            cls=int(raw[rep][2]), group=class_group(raw[rep][2]),
            n_cells=len(members),
            raw_cells=tuple((raw[i][3], raw[i][4], raw[i][2]) for i in members),
        ))
    return dets


def measurement_covariance(x: float, y: float, cfg: CleanerConfig) -> np.ndarray:
    """Polar measurement noise rotated into ego Cartesian coordinates."""
    r = math.hypot(x, y)
    if r < 1e-3:
        ur = np.array([1.0, 0.0])
    else:
        ur = np.array([x / r, y / r])
    ut = np.array([-ur[1], ur[0]])
    U = np.column_stack((ur, ut))
    sigma_r = cfg.sigma_range_m + cfg.sigma_range_slope * r
    D = np.diag([sigma_r ** 2, cfg.sigma_tangential_m ** 2])
    return U @ D @ U.T + np.eye(2) * cfg.quantisation_var


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------

def _rot(delta: float) -> np.ndarray:
    c, s = math.cos(delta), math.sin(delta)
    return np.array([[c, s], [-s, c]])


def transition(psi: float, dt: float, damping: float = 1.0) -> np.ndarray:
    """State transition F for state [x, y, wx, wy].

    ``damping`` < 1 pulls w_y toward zero once per step; that is the
    stationary-infrastructure prior (see the module docstring).  It is folded
    into F so the Kalman filter, the RTS smoother and the extrapolator all use
    exactly the same model.
    """
    M = _rot(psi * dt)
    D = np.diag([1.0, float(damping)])
    F = np.zeros((4, 4))
    F[:2, :2] = M
    F[:2, 2:] = M @ D * dt
    F[2:, 2:] = M @ D
    return F


def process_noise(sigma_a: float, dt: float) -> np.ndarray:
    G = np.zeros((4, 2))
    G[:2, :] = np.eye(2) * (0.5 * dt * dt)
    G[2:, :] = np.eye(2) * dt
    return G @ (np.eye(2) * sigma_a ** 2) @ G.T


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

class Track:
    """One hypothesised physical object with continuous sub-cell state."""

    __slots__ = ("tid", "group", "state", "cov", "frames", "obs_frames",
                 "obs_xy", "obs_cls", "first_frame", "last_obs_frame",
                 "hits", "misses", "confirmed", "alive", "history",
                 "is_static", "smoothed", "plausible", "duplicate_of")

    def __init__(self, tid: int, frame: int, det: Detection, cfg: CleanerConfig):
        self.tid = tid
        self.group = det.group
        self.state = np.array([det.x, det.y, 0.0, 0.0])
        self.cov = np.diag([
            cfg.sigma_range_m ** 2 + cfg.quantisation_var,
            cfg.sigma_tangential_m ** 2 + cfg.quantisation_var,
            cfg.sigma_v0 ** 2, cfg.sigma_v0 ** 2,
        ])
        self.first_frame = frame
        self.last_obs_frame = frame
        self.obs_frames: List[int] = [frame]
        self.obs_xy: List[Tuple[float, float]] = [(det.x, det.y)]
        self.obs_cls: List[int] = [det.cls]
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.alive = True
        self.is_static = False
        self.plausible = True
        self.duplicate_of: Optional[int] = None
        # history[frame] = (x, y, wx, wy, observed_bool)
        self.history: Dict[int, Tuple[float, float, float, float, bool]] = {}
        self.smoothed: Dict[int, Tuple[float, float, float, float]] = {}

    # -- convenience -------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.state[:2]

    @property
    def n_obs(self) -> int:
        return len(self.obs_frames)

    def predict(self, psi: float, dt: float, sigma_a: float,
                damping: float = 1.0) -> None:
        F = transition(psi, dt, damping)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + process_noise(sigma_a, dt)

    def update(self, det: Detection, cfg: CleanerConfig) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = measurement_covariance(self.state[0], self.state[1], cfg)
        z = np.array([det.x, det.y])
        innov = z - H @ self.state
        S = H @ self.cov @ H.T + R
        d2 = float(innov @ np.linalg.solve(S, innov))
        # M-estimator style downweighting of heavy-tailed range outliers.
        if d2 > cfg.robust_chi2:
            R = R * (d2 / cfg.robust_chi2)
            S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innov
        I_KH = np.eye(4) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + K @ R @ K.T

    def voted_class(self) -> int:
        """Stable label for the track.

        Stateful groups (traffic lights) follow the latest observation because
        their class genuinely changes.  Everything else uses a majority vote,
        which is what removes the measured car<->truck flicker.
        """
        if not self.obs_cls:
            return 0
        if self.group in STATEFUL_GROUPS:
            return int(self.obs_cls[-1])
        counts: Dict[int, int] = {}
        for c in self.obs_cls:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts.values())
        tied = [c for c, n in counts.items() if n == best]
        if len(tied) == 1:
            return int(tied[0])
        for c in reversed(self.obs_cls):      # tie-break: most recent
            if c in tied:
                return int(c)
        return int(min(tied))

    def class_at(self, frame: int) -> int:
        """Label emitted at ``frame``.

        For stateful groups this is the most recent observation at or before
        the frame, so a red->green transition is not back-propagated.
        """
        if self.group not in STATEFUL_GROUPS:
            return self.voted_class()
        cls = self.obs_cls[0]
        for f, c in zip(self.obs_frames, self.obs_cls):
            if f <= frame:
                cls = c
            else:
                break
        return int(cls)


def lateral_damping(group: str, cfg: CleanerConfig, yaw_compensated: bool,
                    dt: float = DT) -> float:
    """Per-step decay applied to w_y for stationary-infrastructure classes.

    Only valid once yaw is compensated: without it a static object seen during
    a turn genuinely has w_y = -psi*x, and damping would fight the turn.
    Expressed as a time constant, so it is unchanged by the frame rate.
    """
    if not yaw_compensated or group not in cfg.static_prior_groups:
        return 1.0
    return float(math.exp(-dt / max(cfg.tau_lateral_s, 1e-3)))


def _radial_tangential_error(p: Tuple[float, float],
                             q: Tuple[float, float]) -> Tuple[float, float]:
    """Split the p->q offset into along-the-ray and across-the-ray parts."""
    r = math.hypot(*p)
    ur = np.array([1.0, 0.0]) if r < 1e-3 else np.array([p[0] / r, p[1] / r])
    ut = np.array([-ur[1], ur[0]])
    d = np.array([q[0] - p[0], q[1] - p[1]])
    return abs(float(d @ ur)), abs(float(d @ ut))


def _inside_grid(x: float, y: float, margin: float) -> bool:
    return (X_MIN - margin <= x <= X_MAX + margin) and (Y_MIN - margin <= y <= Y_MAX + margin)


# ---------------------------------------------------------------------------
# Tracker (one forward pass)
# ---------------------------------------------------------------------------

class Tracker:
    """Global-nearest-neighbour Kalman tracker driven by estimated ego motion."""

    def __init__(self, cfg: CleanerConfig,
                 ego_v: Optional[np.ndarray] = None,
                 ego_psi: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.ego_v = ego_v
        self.ego_psi = ego_psi
        self.active: List[Track] = []
        self.finished: List[Track] = []
        self._next_id = 0

    @property
    def yaw_ok(self) -> bool:
        return bool(self.cfg.use_ego_motion and self.ego_psi is not None)

    def _yaw(self, frame: int) -> float:
        """Estimated ego yaw rate at ``frame`` (the only ego input to the model)."""
        if not self.yaw_ok:
            return 0.0
        psi = float(self.ego_psi[min(max(frame, 0), len(self.ego_psi) - 1)])
        return psi if np.isfinite(psi) else 0.0

    def step(self, frame: int, dets: Sequence[Detection],
             dt: float = DT, psi: Optional[float] = None) -> None:
        """Advance the tracker by one frame.

        ``psi`` overrides the stored yaw-rate series; this is how a live
        pipeline injects a measured yaw rate.  ``dt`` lets the same tracker run
        at a rate other than 10 FPS: everything below that is expressed
        per-frame is scaled by dt/DT so behaviour at dt = DT is unchanged.
        """
        cfg = self.cfg
        if psi is None:
            psi = self._yaw(frame)
        else:
            psi = float(psi) if np.isfinite(psi) else 0.0
        rate = dt / DT   # 1.0 for this dataset

        for t in self.active:
            sa = cfg.sigma_accel.get(t.group, 2.0)
            t.predict(psi, dt, sa,
                      lateral_damping(t.group, cfg, self.yaw_ok or psi != 0.0, dt))

        # ---- association -------------------------------------------------
        n_t, n_d = len(self.active), len(dets)
        if n_t and n_d:
            BIG = 1e6
            cost = np.full((n_t, n_d), BIG)
            for i, t in enumerate(self.active):
                coast = (frame - t.last_obs_frame) * rate
                rng = math.hypot(t.state[0], t.state[1])
                gate_r = (cfg.gate_max_m + cfg.gate_range_frac * rng
                          + cfg.gate_grow_m_per_frame * coast)
                gate_t = (cfg.gate_tangential_m + cfg.gate_tangential_range_frac * rng
                          + cfg.gate_tangential_per_frame * coast)
                H = np.zeros((2, 4)); H[0, 0] = 1.0; H[1, 1] = 1.0
                R = measurement_covariance(t.state[0], t.state[1], cfg)
                S = H @ t.cov @ H.T + R
                Sinv = np.linalg.inv(S)
                logdet = float(np.log(max(np.linalg.det(S), 1e-9)))
                for j, d in enumerate(dets):
                    if d.group != t.group:
                        continue
                    dx = d.x - t.state[0]
                    dy = d.y - t.state[1]
                    er, et = _radial_tangential_error((t.state[0], t.state[1]),
                                                      (d.x, d.y))
                    if er > gate_r or et > gate_t:
                        continue
                    innov = np.array([dx, dy])
                    d2 = float(innov @ Sinv @ innov)
                    if d2 > cfg.gate_chi2:
                        continue
                    c = d2 + logdet
                    if t.obs_cls and d.cls != t.obs_cls[-1]:
                        c += cfg.class_switch_penalty
                    cost[i, j] = c
            rows, colsx = linear_sum_assignment(cost)
            assigned_t, assigned_d = set(), set()
            for i, j in zip(rows, colsx):
                if cost[i, j] >= BIG / 2:
                    continue
                t = self.active[i]
                t.update(dets[j], cfg)
                t.obs_frames.append(frame)
                t.obs_xy.append((dets[j].x, dets[j].y))
                t.obs_cls.append(dets[j].cls)
                t.last_obs_frame = frame
                t.hits += 1
                t.misses = 0
                assigned_t.add(i)
                assigned_d.add(j)
        else:
            assigned_t, assigned_d = set(), set()

        for i, t in enumerate(self.active):
            if i not in assigned_t:
                t.misses += 1

        # ---- births ------------------------------------------------------
        for j, d in enumerate(dets):
            if j in assigned_d:
                continue
            t = Track(self._next_id, frame, d, cfg)
            self._next_id += 1
            self.active.append(t)

        # ---- confirmation, history, death --------------------------------
        keep: List[Track] = []
        for t in self.active:
            if not t.confirmed:
                window = frame - t.first_frame + 1
                if t.hits >= cfg.confirm_hits:
                    t.confirmed = True
                elif window > cfg.confirm_window:
                    # Never confirmed in time: let it die unless still hitting.
                    if t.misses > 0:
                        t.alive = False
            observed = (t.last_obs_frame == frame)
            t.history[frame] = (float(t.state[0]), float(t.state[1]),
                                float(t.state[2]), float(t.state[3]), observed)

            max_coast = cfg.max_coast_frames.get(t.group, 10) / max(rate, 1e-6)
            pos_sigma = math.sqrt(max(float(t.cov[0, 0] + t.cov[1, 1]), 0.0))
            if t.misses > max_coast:
                t.alive = False
            if not _inside_grid(t.state[0], t.state[1], cfg.grid_margin_m):
                t.alive = False
            if pos_sigma > cfg.max_pos_sigma_m and t.misses > 0:
                t.alive = False

            if t.alive:
                keep.append(t)
            else:
                self.finished.append(t)
        self.active = keep

    def finish(self) -> List[Track]:
        out = self.finished + self.active
        self.finished, self.active = [], []
        out.sort(key=lambda t: (t.first_frame, t.tid))
        return out


# ---------------------------------------------------------------------------
# Ego-motion estimation
# ---------------------------------------------------------------------------

class ScalarCVFilter:
    """Causal constant-rate Kalman filter on a scalar signal.

    This is exactly the FORWARD half of _rts_smooth_1d, so the online ego-motion
    estimate obeys the same dynamics model as the offline one (same measurement
    noise, same rate-process noise) -- it simply cannot use the backward pass.
    A fixed-gain exponential filter was used here before; it both attenuated and
    delayed turns because its time constant is unrelated to vehicle dynamics.
    """

    def __init__(self, sigma_meas: float, sigma_rate: float, dt: float,
                 x0: float = 0.0, p0: float = 25.0, rate_damping: float = 1.0):
        """``rate_damping`` < 1 stops the rate state from extrapolating a turn
        through the long measurement gaps this data has (only a minority of
        frames yield an ego solve).  1.0 is the constant-rate model the offline
        smoother uses; 0.0 degenerates to a random walk."""
        self.dt = dt
        d = float(rate_damping)
        self.F = np.array([[1.0, dt * d], [0.0, d]])
        self.Q = np.array([[dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 2 / 2, dt]]) * sigma_rate ** 2
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[sigma_meas ** 2]])
        self.x = np.array([x0, 0.0])
        self.P = np.diag([p0, p0])
        self.seeded = False

    def step(self, z: Optional[float]) -> float:
        """Advance one frame; ``z`` is None when this frame has no measurement."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if z is not None:
            if not self.seeded:
                self.x = np.array([float(z), 0.0])
                self.seeded = True
            else:
                y = np.array([float(z)]) - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + (K @ y).ravel()
                self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0])


def _rts_smooth_1d(z: np.ndarray, valid: np.ndarray, sigma_meas: float,
                   sigma_rate: float, dt: float,
                   x0: float = 0.0) -> np.ndarray:
    """Constant-rate (position+rate) RTS smoother on a scalar signal."""
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]) * sigma_rate ** 2
    H = np.array([[1.0, 0.0]])
    R = np.array([[sigma_meas ** 2]])

    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    x = np.array([x0, 0.0]); P = np.diag([25.0, 25.0])
    for k in range(n):
        x = F @ x; P = F @ P @ F.T + Q
        xp[k] = x; Pp[k] = P
        if valid[k]:
            y = np.array([z[k]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ y).ravel()
            P = (np.eye(2) - K @ H) @ P
        xf[k] = x; Pf[k] = P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0]


def solve_ego_frame(points: Sequence[Tuple[float, float, float, float, float]],
                    cfg: CleanerConfig,
                    rs: np.random.RandomState) -> Optional[Tuple[float, float, int]]:
    """Recover (v, psi) from one frame's velocity field.  Returns None if unsolvable.

    A world-static point at ego coords (x, y) has ego-relative velocity
        vx = -v + psi*y ,   vy = -psi*x
    so two static tracks determine (v, psi).  RANSAC picks the static subset;
    vehicles moving with the ego are the outliers.  Each point carries an
    ``infra`` flag marking classes that are certainly stationary.

    Used identically by the offline batch estimator and the online cleaner, so
    there is exactly one implementation of the ego-motion maths.
    """
    if len(points) < 2:
        return None
    P = np.asarray(points, dtype=float)
    best_inl, best_n = None, -1
    idx = np.arange(len(P))
    iters = min(cfg.ego_ransac_iters, max(1, len(P) * (len(P) - 1) // 2 * 3))
    for _ in range(iters):
        sel = rs.choice(idx, 2, replace=False)
        A, b = [], []
        for x, y, vx, vy, _infra in P[sel]:
            A.append([-1.0, y]); b.append(vx)
            A.append([0.0, -x]); b.append(vy)
        try:
            sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        except np.linalg.LinAlgError:
            continue
        v, psi = float(sol[0]), float(sol[1])
        if not (-1.0 <= v <= cfg.ego_v_max and abs(psi) <= cfg.ego_psi_max):
            continue
        res = np.hypot(P[:, 2] - (-v + psi * P[:, 1]), P[:, 3] - (-psi * P[:, 0]))
        inl = res < cfg.ego_inlier_tol_mps
        k = int(inl.sum())
        if k > best_n:
            best_n, best_inl = k, inl
    if best_inl is None or best_n < 2:
        return None
    # Refine: yaw rate from the LATERAL flow only (bearing is the accurate
    # measurement), forward speed from certainly-static infrastructure when it
    # is present, otherwise from NEAR points only (the range bias grows with
    # range, so distant points would inflate v).
    Q = P[best_inl]
    denom = float(np.sum(Q[:, 0] ** 2))
    psi = float(-np.sum(Q[:, 3] * Q[:, 0]) / denom) if denom > 1e-6 else 0.0
    infra = Q[:, 4] > 0.5
    if infra.any():
        src = Q[infra]
    else:
        near = np.hypot(Q[:, 0], Q[:, 1]) <= cfg.ego_near_range_m
        src = Q[near] if near.sum() >= 1 else Q
    v = float(np.median(-(src[:, 2] - psi * src[:, 1])))
    if not (-1.0 <= v <= cfg.ego_v_max and abs(psi) <= cfg.ego_psi_max):
        return None
    return v, psi, best_n


def _finite_difference_velocity(t: Track, frame: int, cfg: CleanerConfig
                                ) -> Optional[Tuple[float, float, float, float]]:
    """Velocity of a track at ``frame`` straight from its own observations.

    The ego-motion solver reads the bearing RATE, and a Kalman velocity state is
    a low-pass of exactly that signal: measured on this dataset it attenuates
    the yaw rate by ~1.8x through a turn and delays it by several frames.  So
    the solver is given a finite difference instead, over a baseline of at least
    ego_flow_baseline_frames -- long enough to span one 5 Hz perception update
    so the measured hold-and-jump averages out rather than aliasing.

    Uses only observations at or before ``frame``, so it is causal.
    """
    obs = {}
    for f, xy in zip(t.obs_frames, t.obs_xy):
        obs[f] = xy
    if frame not in obs:
        return None
    for gap in range(cfg.ego_flow_baseline_frames,
                     cfg.ego_flow_max_baseline_frames + 1):
        f0 = frame - gap
        if f0 in obs:
            x1, y1 = obs[frame]
            x0, y0 = obs[f0]
            span = gap * DT
            return (x1, y1, (x1 - x0) / span, (y1 - y0) / span)
    return None


def ego_points_from_tracks(tracks: Sequence[Track], frame: int,
                           cfg: CleanerConfig,
                           causal: bool) -> List[Tuple[float, float, float, float, float]]:
    """Velocity field at ``frame`` from a bootstrap tracker's tracks.

    Prefers finite-difference velocities (see _finite_difference_velocity); the
    filtered state is used only as a coverage fallback when fewer than two
    tracks have a usable finite difference this frame, so every individual
    solve is internally consistent.
    """
    fd_pts: List[Tuple[float, float, float, float, float]] = []
    kf_pts: List[Tuple[float, float, float, float, float]] = []
    for t in tracks:
        h = t.history.get(frame)
        if h is None:
            continue
        if causal:
            # Live mode may only use the support this track had BY frame f.
            if int(np.searchsorted(sorted(t.obs_frames), frame, side="right")) < cfg.confirm_hits:
                continue
        elif t.n_obs < cfg.confirm_hits:
            continue
        # Infrastructure classes are certainly stationary, so they are the
        # trustworthy source for ego speed; everything else may be moving with
        # the ego and would bias v low.
        infra = 1.0 if t.group in cfg.static_prior_groups else 0.0
        kf_pts.append((h[0], h[1], h[2], h[3], infra))
        fd = (_finite_difference_velocity(t, frame, cfg)
              if cfg.ego_flow_finite_difference else None)
        if fd is not None:
            fd_pts.append((fd[0], fd[1], fd[2], fd[3], infra))
    return fd_pts if len(fd_pts) >= 2 else kf_pts


def estimate_ego_motion(tracks: Sequence[Track], n_frames: int,
                        cfg: CleanerConfig,
                        causal: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline batch ego-motion estimate: per-frame solve, then RTS smoothing.

    In the bootstrap pass ego motion is zero, so each track's (wx, wy) is the
    measured ego-relative velocity, which is what the static-point model reads.
    """
    v_raw = np.full(n_frames, np.nan)
    psi_raw = np.full(n_frames, np.nan)
    n_in = np.zeros(n_frames, dtype=int)
    rs = np.random.RandomState(0)   # deterministic

    for f in range(n_frames):
        r = solve_ego_frame(ego_points_from_tracks(tracks, f, cfg, causal), cfg, rs)
        if r is None:
            continue
        v_raw[f], psi_raw[f], n_in[f] = r

    valid = np.isfinite(v_raw) & (n_in >= cfg.ego_min_inliers)
    if valid.sum() < 5:
        return (np.zeros(n_frames), np.zeros(n_frames), n_in)

    v_fill = np.where(valid, v_raw, 0.0)
    psi_fill = np.where(valid, psi_raw, 0.0)
    v_s = _rts_smooth_1d(v_fill, valid, cfg.ego_sigma_v_meas,
                         cfg.ego_sigma_a, DT,
                         x0=float(np.nanmedian(v_raw[valid])))
    psi_s = _rts_smooth_1d(psi_fill, valid, cfg.ego_sigma_psi_meas,
                           cfg.ego_sigma_psidot, DT, x0=0.0)
    v_s = np.clip(v_s, 0.0, cfg.ego_v_max)
    psi_s = np.clip(psi_s, -cfg.ego_psi_max, cfg.ego_psi_max)
    return v_s, psi_s, n_in


# ---------------------------------------------------------------------------
# Offline refinement: stitching, merging, smoothing
# ---------------------------------------------------------------------------

def _refilter(track: Track, cfg: CleanerConfig, ego_psi: np.ndarray,
              yaw_ok: bool = True) -> Track:
    """Re-run the Kalman filter and the RTS smoother over a track's observations.

    Called after any operation that changes the observation list (stitching,
    duplicate merging) so the emitted trajectory is always the smoothed fit to
    the full, final evidence for that object.
    """
    obs: Dict[int, Tuple[Tuple[float, float], int]] = {}
    for f, xy, c in zip(track.obs_frames, track.obs_xy, track.obs_cls):
        obs[f] = (xy, c)
    f0, f1 = min(obs), max(obs)

    sa = cfg.sigma_accel.get(track.group, 2.0)
    damp = lateral_damping(track.group, cfg, yaw_ok)

    x = np.array([obs[f0][0][0], obs[f0][0][1], 0.0, 0.0])
    P = np.diag([cfg.sigma_range_m ** 2, cfg.sigma_tangential_m ** 2,
                 cfg.sigma_v0 ** 2, cfg.sigma_v0 ** 2])

    n = f1 - f0 + 1
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    Fs = np.zeros((n, 4, 4))

    for k in range(n):
        f = f0 + k
        psi = float(ego_psi[min(max(f, 0), len(ego_psi) - 1)]) if yaw_ok else 0.0
        if k == 0:
            F = np.eye(4); Q = np.zeros((4, 4))
        else:
            F = transition(psi, DT, damp)
            Q = process_noise(sa, DT)
        x = F @ x
        P = F @ P @ F.T + Q
        Fs[k] = F
        xp[k] = x; Pp[k] = P

        if f in obs:
            (mx, my), _c = obs[f]
            H = np.zeros((2, 4)); H[0, 0] = 1.0; H[1, 1] = 1.0
            R = measurement_covariance(x[0], x[1], cfg)
            innov = np.array([mx, my]) - H @ x
            S = H @ P @ H.T + R
            d2 = float(innov @ np.linalg.solve(S, innov))
            if d2 > cfg.robust_chi2:
                R = R * (d2 / cfg.robust_chi2)
                S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ innov
            IKH = np.eye(4) - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T

        xf[k] = x; Pf[k] = P

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        try:
            C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        except np.linalg.LinAlgError:
            continue
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])

    track.smoothed = {f0 + k: (float(xs[k, 0]), float(xs[k, 1]),
                               float(xs[k, 2]), float(xs[k, 3]))
                      for k in range(n)}
    track.first_frame = f0
    track.last_obs_frame = f1
    return track


def _extrapolate(track: Track, frame: int, ego_psi: np.ndarray,
                 cfg: CleanerConfig, yaw_ok: bool = True,
                 max_steps: int = 60) -> Optional[Tuple[float, float]]:
    """Propagate a smoothed track state to a frame outside its observed span."""
    if not track.smoothed:
        return None
    fs = sorted(track.smoothed)
    if fs[0] <= frame <= fs[-1]:
        s = track.smoothed[frame]
        return (s[0], s[1])
    anchor, step = (fs[0], -1) if frame < fs[0] else (fs[-1], 1)
    damp = lateral_damping(track.group, cfg, yaw_ok)
    x = np.array(track.smoothed[anchor])
    f = anchor
    while f != frame and abs(f - anchor) < max_steps:
        psi = float(ego_psi[min(max(f, 0), len(ego_psi) - 1)]) if yaw_ok else 0.0
        x = transition(psi, DT * step, damp if step > 0 else 1.0) @ x
        f += step
    return (float(x[0]), float(x[1]))


def _fit_rms(track: Track) -> float:
    """RMS distance between a track's own measurements and its smoothed fit."""
    if not track.smoothed:
        return float("inf")
    d = [math.hypot(xy[0] - track.smoothed[f][0], xy[1] - track.smoothed[f][1])
         for f, xy in zip(track.obs_frames, track.obs_xy) if f in track.smoothed]
    return float(np.sqrt(np.mean(np.square(d)))) if d else float("inf")


def _try_combine(a: Track, b: Track, cfg: CleanerConfig, ego_psi: np.ndarray,
                 yaw_ok: bool) -> bool:
    """Tentatively fold b into a; keep it only if the combined fit stays good.

    This is the hypothesis test that separates "one object seen twice" from
    "two objects that happen to be close".  On rejection a is restored exactly.
    """
    keep = (list(a.obs_frames), list(a.obs_xy), list(a.obs_cls),
            dict(a.smoothed), a.first_frame, a.last_obs_frame, a.hits)
    rms_before = max(_fit_rms(a), _fit_rms(b))

    frames = a.obs_frames + b.obs_frames
    xy = a.obs_xy + b.obs_xy
    cls = a.obs_cls + b.obs_cls
    order = sorted(range(len(frames)), key=lambda k: (frames[k], k))
    a.obs_frames = [frames[k] for k in order]
    a.obs_xy = [xy[k] for k in order]
    a.obs_cls = [cls[k] for k in order]
    a.hits = len(a.obs_frames)
    _refilter(a, cfg, ego_psi, yaw_ok)

    rms_after = _fit_rms(a)
    ok = (rms_after <= cfg.max_fit_rms_m
          and rms_after <= max(rms_before, 0.5) * cfg.fit_degradation_factor + 0.5)
    if not ok:
        (a.obs_frames, a.obs_xy, a.obs_cls, a.smoothed,
         a.first_frame, a.last_obs_frame, a.hits) = keep
    return ok


def merge_duplicate_tracks(tracks: List[Track], cfg: CleanerConfig,
                           ego_psi: np.ndarray, yaw_ok: bool = True) -> List[Track]:
    """Merge tracks that are two views of one physical object.

    Two tracks are merged when they belong to the same class group, coexist for
    at least ``merge_min_overlap_frames`` frames, and stay within a
    group-specific radius throughout.  For static groups (signs, lights) the
    radius is larger because the measured duplicate separation is 3-8 m while
    genuine opposite-corner sign pairs are ~35 m apart.
    """
    changed = True
    guard = 0
    while changed and guard < 10:
        changed = False
        guard += 1
        tracks.sort(key=lambda t: (t.first_frame, t.tid))
        merged_into: Dict[int, int] = {}
        by_id = {t.tid: t for t in tracks}
        for i in range(len(tracks)):
            a = tracks[i]
            if a.tid in merged_into:
                continue
            for j in range(i + 1, len(tracks)):
                b = tracks[j]
                if b.tid in merged_into or b.group != a.group:
                    continue
                fa = set(a.smoothed); fb = set(b.smoothed)
                common = sorted(fa & fb)
                if len(common) < cfg.merge_min_overlap_frames:
                    continue
                d = [math.hypot(a.smoothed[f][0] - b.smoothed[f][0],
                                a.smoothed[f][1] - b.smoothed[f][1])
                     for f in common]
                rad = cfg.merge_radius_m.get(a.group, 3.0)
                if float(np.median(d)) > rad or float(np.percentile(d, 90)) > rad * 1.6:
                    continue
                if _try_combine(a, b, cfg, ego_psi, yaw_ok):
                    merged_into[b.tid] = a.tid
                    changed = True
        if merged_into:
            tracks = [t for t in tracks if t.tid not in merged_into]
    return tracks


def stitch_tracklets(tracks: List[Track], cfg: CleanerConfig,
                     ego_psi: np.ndarray, yaw_ok: bool = True) -> List[Track]:
    """Link a dying tracklet to a later one that is the same physical object.

    OFFLINE ONLY: it needs frames after the tracklet ended.  This is what
    recovers identity across dropouts longer than the coasting budget.
    """
    tracks.sort(key=lambda t: (t.first_frame, t.tid))
    ends = [t for t in tracks if t.smoothed and t.n_obs >= cfg.stitch_min_obs]
    if len(ends) < 2:
        return tracks
    BIG = 1e6
    n = len(ends)
    cost = np.full((n, n), BIG)
    for i, a in enumerate(ends):
        a_end = max(a.smoothed)
        for j, b in enumerate(ends):
            if i == j or b.group != a.group:
                continue
            b_start = min(b.smoothed)
            gap = b_start - a_end
            if gap <= 0 or gap > cfg.stitch_max_gap_frames:
                continue
            pred = _extrapolate(a, b_start, ego_psi, cfg, yaw_ok)
            back = _extrapolate(b, a_end, ego_psi, cfg, yaw_ok)
            if pred is None or back is None:
                continue
            e1 = _radial_tangential_error(pred, b.smoothed[b_start][:2])
            e2 = _radial_tangential_error(back, a.smoothed[a_end][:2])
            r_ref = 0.5 * (math.hypot(*pred) + math.hypot(*back))
            gate_r = (cfg.stitch_gate_radial_m + cfg.stitch_gate_radial_frac * r_ref
                      + cfg.stitch_gate_per_frame_m * gap)
            gate_t = cfg.stitch_gate_tangential_m + 0.3 * gap
            if max(e1[0], e2[0]) > gate_r or max(e1[1], e2[1]) > gate_t:
                continue
            cost[i, j] = (max(e1[0], e2[0]) / gate_r
                          + 2.0 * max(e1[1], e2[1]) / gate_t + 0.02 * gap)

    rows, colsx = linear_sum_assignment(cost)
    link: Dict[int, int] = {}
    for i, j in zip(rows, colsx):
        if cost[i, j] < BIG / 2:
            link[i] = j

    # Follow chains so A->B->C collapses into one track.
    absorbed = set()
    for i in sorted(link):
        if i in absorbed:
            continue
        a = ends[i]
        j = link.get(i)
        while j is not None and j not in absorbed and j != i:
            if not _try_combine(a, ends[j], cfg, ego_psi, yaw_ok):
                break
            absorbed.add(j)
            j = link.get(j)
    kept = [t for k, t in enumerate(ends) if k not in absorbed]
    short = [t for t in tracks if not (t.smoothed and t.n_obs >= cfg.stitch_min_obs)]
    return sorted(kept + short, key=lambda t: (t.first_frame, t.tid))


def suppress_duplicates(tracks: Sequence[Track], cfg: CleanerConfig) -> None:
    """Flag tracks that are a redundant second view of a better-supported one.

    Runs after merging, so it only sees pairs the likelihood test refused to
    combine -- typically one physical stop sign split into two range hypotheses
    that cannot be explained by a single smooth trajectory.  The survivor is the
    one with more evidence; ties go to the nearer track, which is the
    conservative choice for the target-speed policy.
    """
    order = sorted(tracks, key=lambda t: (-t.n_obs, t.tid))
    for i, a in enumerate(order):
        if a.duplicate_of is not None or not a.smoothed:
            continue
        for b in order[i + 1:]:
            if b.duplicate_of is not None or b.group != a.group or not b.smoothed:
                continue
            if not (a.n_obs >= cfg.duplicate_min_support
                    or a.n_obs >= cfg.duplicate_support_ratio * b.n_obs):
                continue
            common = sorted(set(a.smoothed) & set(b.smoothed))
            if len(common) < cfg.duplicate_min_overlap_frames:
                continue
            d = [math.hypot(a.smoothed[f][0] - b.smoothed[f][0],
                            a.smoothed[f][1] - b.smoothed[f][1]) for f in common]
            rad = cfg.duplicate_radius_m.get(a.group, 3.0)
            if float(np.median(d)) > rad:
                continue
            if a.n_obs == b.n_obs:
                ra = np.median([math.hypot(*a.smoothed[f][:2]) for f in common])
                rb = np.median([math.hypot(*b.smoothed[f][:2]) for f in common])
                if rb < ra:
                    a, b = b, a
            b.duplicate_of = a.tid


def flag_plausibility(tracks: Sequence[Track], cfg: CleanerConfig) -> None:
    """Mark tracks whose recovered motion is not physically possible."""
    for t in tracks:
        if not t.smoothed:
            t.plausible = False
            continue
        obs = set(t.obs_frames)
        wy = [abs(s[3]) for f, s in t.smoothed.items() if f in obs]
        wx = [abs(s[2]) for f, s in t.smoothed.items() if f in obs]
        cap = cfg.max_lateral_world_speed_mps.get(t.group, 8.0)
        t.plausible = bool(wy and np.median(wy) <= cap
                           and np.median(wx) <= cfg.max_radial_world_speed_mps)


def classify_static(tracks: Sequence[Track], cfg: CleanerConfig,
                    ego_v: np.ndarray) -> None:
    """Annotate tracks that look like stationary world objects.

    Reported for visualisation and diagnostics only -- it does not drive the
    filter.  A stationary object has w = (-v_ego, 0) in the yaw-compensated
    ego frame, so it is flagged by comparing w against the estimated ego speed.
    Infrastructure classes are flagged unconditionally: that is their prior.
    """
    for t in tracks:
        if t.group in cfg.static_prior_groups:
            t.is_static = True
            continue
        if not t.smoothed:
            t.is_static = False
            continue
        obs = set(t.obs_frames)
        res = [math.hypot(s[2] + float(ego_v[min(f, len(ego_v) - 1)]), s[3])
               for f, s in t.smoothed.items() if f in obs]
        t.is_static = bool(res and np.median(res) < cfg.static_speed_mps)


def _in_safety_zone(x: float, y: float, cls: int, cfg: CleanerConfig) -> bool:
    """True if a detection here could influence getTargetSpeed."""
    if x <= 0.0 or x > cfg.safety_max_forward_m:
        return False
    if cls in TS_OBSTACLE_CLASSES:
        return abs(y) <= cfg.safety_obstacle_half_width_m
    if cls in TS_STOP_CLASSES:
        return abs(y) <= cfg.safety_stop_half_width_m
    return False


def build_emissions(tracks: Sequence[Track], n_frames: int,
                    cfg: CleanerConfig,
                    ego_psi: Optional[np.ndarray] = None,
                    causal: bool = False,
                    yaw_ok: bool = True) -> List[List[dict]]:
    """Decide, per frame, which tracks are written out and where.

    Asymmetric by design:
      * a CONFIRMED track is emitted over its whole observed span plus up to
        ``emit_coast_frames`` afterwards -- this is what repairs dropouts;
      * an UNCONFIRMED track is emitted only on frames where it actually has a
        detection AND that detection lies in the control-relevant safety zone
        -- so a pedestrian appearing 8 m ahead is never delayed, while far
        clutter is discarded.
    """
    if ego_psi is None:
        ego_psi = np.zeros(n_frames)

    def is_established(t: Track, f: int) -> bool:
        if not t.plausible or t.duplicate_of is not None:
            return False
        if causal:
            return int(np.searchsorted(t.obs_frames, f, side="right")) >= cfg.min_track_observations
        return t.n_obs >= cfg.min_track_observations

    out: List[List[dict]] = [[] for _ in range(n_frames)]
    for t in tracks:
        if not t.smoothed:
            continue
        obs = set(t.obs_frames)
        f_first, f_last = min(t.smoothed), max(t.smoothed)
        trail = cfg.emit_coast_frames if causal else 0
        f_stop = min(n_frames - 1, f_last + trail)
        for f in range(f_first, f_stop + 1):
            if f in t.smoothed:
                x, y, wx, wy = t.smoothed[f]
            else:
                p = _extrapolate(t, f, ego_psi, cfg, yaw_ok)
                if p is None:
                    continue
                x, y = p
                wx = wy = 0.0
            cls = t.class_at(f)
            if not is_established(t, f):
                # Not enough evidence for this to be its own object.  Anything
                # control-relevant that this drops is restored verbatim by
                # add_safety_passthrough() below.
                continue
            row = EGO_ROW - int(round(x))
            col = EGO_COL + int(round(y))
            if not (0 <= row < ROWS and 0 <= col < COLS):
                continue
            if f in obs:
                prov = PROV_OBSERVED
            elif f <= f_last:
                prov = PROV_INTERPOLATED
            else:
                prov = PROV_COASTED
            out[f].append(dict(tid=t.tid, cls=cls, x=x, y=y, wx=wx, wy=wy,
                               row=row, col=col, observed=(f in obs),
                               static=t.is_static, group=t.group,
                               n_obs=t.n_obs, provenance=prov))
    return out


def add_safety_passthrough(emissions: List[List[dict]],
                           dets_per_frame: Sequence[Sequence[Detection]],
                           cfg: CleanerConfig) -> int:
    """Restore any control-relevant raw detection the tracker did not represent.

    This enforces the safety invariant of the whole pipeline:

        for every raw detection inside the region getTargetSpeed can react to,
        the cleaned frame contains a detection of the same class group within
        duplicate_radius_m of it.

    Cleaning may therefore reject clutter, repair dropouts and re-estimate
    positions, but it can never be the reason a close obstacle vanishes -- not
    because the track was unconfirmed, and not because a robust filter update
    treated the measurement as an outlier.  Redundant detections (already
    covered by an emitted object of the same group) are the one exception, and
    that is what removes duplicate stop signs rather than losing evidence.

    Returns the number of restored detections.
    """
    restored = 0
    for f, dets in enumerate(dets_per_frame):
        ems = emissions[f]
        for d in dets:
            if not _in_safety_zone(d.x, d.y, d.cls, cfg):
                continue
            rad = cfg.passthrough_cover_radius_m.get(d.group, 3.0)
            if any(e["group"] == d.group
                   and math.hypot(e["x"] - d.x, e["y"] - d.y) <= rad for e in ems):
                continue
            row = EGO_ROW - int(round(d.x))
            col = EGO_COL + int(round(d.y))
            if not (0 <= row < ROWS and 0 <= col < COLS):
                continue
            ems.append(dict(tid=-1, cls=d.cls, x=d.x, y=d.y, wx=0.0, wy=0.0,
                            row=row, col=col, observed=True, static=False,
                            group=d.group, n_obs=0,
                            provenance=PROV_PASSTHROUGH))
            restored += 1
    return restored


def rasterise(emissions: Sequence[dict]) -> np.ndarray:
    """Emissions of one frame -> a 120x80 uint8 matrix on the original grid.

    Cell conflicts resolve to the class with the highest braking demand, so a
    collision can never relax the target speed.
    """
    m = np.zeros((ROWS, COLS), dtype=np.uint8)
    for e in sorted(emissions, key=lambda e: (_priority(e["cls"]), -e["n_obs"], -e["tid"])):
        m[e["row"], e["col"]] = np.uint8(e["cls"])
    return m


def rasterise_hires(emissions: Sequence[dict], scale: int = 4) -> np.ndarray:
    """Optional finer raster for VISUALISATION/ANALYSIS ONLY.

    Same physical extent and same origin; ``scale`` sub-cells per metre.  This
    is never fed to getTargetSpeed -- see the module docstring and the report.
    """
    m = np.zeros((ROWS * scale, COLS * scale), dtype=np.uint8)
    for e in sorted(emissions, key=lambda e: (_priority(e["cls"]), -e["n_obs"], -e["tid"])):
        r = int(round((EGO_ROW - e["x"]) * scale))
        c = int(round((e["y"] + EGO_COL) * scale))
        if 0 <= r < m.shape[0] and 0 <= c < m.shape[1]:
            m[r, c] = np.uint8(e["cls"])
    return m


# ---------------------------------------------------------------------------
# Online / causal streaming interface
# ---------------------------------------------------------------------------

class OnlineTemporalCleaner:
    """Stateful, strictly causal temporal cleaner for a live 10 FPS pipeline.

    Usage::

        cleaner = OnlineTemporalCleaner()
        for matrix in stream:                       # one call per frame
            cleaned, objects = cleaner.update(matrix, dt=0.10)
            speed = getTargetSpeed(matrix=cleaned, ...)

    Each ``update`` sees only the current matrix, the state accumulated from
    previous frames, and ego motion available now.  It never touches a future
    matrix, future track support, backward smoothing or retroactive stitching.

    It uses the SAME tracker as the offline pipeline -- the same Kalman model,
    the same anisotropic gates, the same Hungarian association, the same class
    voting and the same safety pass-through.  Only the four offline-only stages
    are replaced:

    ==========================  ==================================================
    offline stage               causal replacement
    ==========================  ==================================================
    RTS backward smoothing      the forward Kalman posterior (lags by ~1 frame,
                                and cannot un-see an outlier it already absorbed)
    interior-gap interpolation  forward coasting for up to emit_coast_frames;
                                the gap is filled as it happens, not afterwards
    tracklet stitching          the coasting budget plus the anisotropic gate;
                                gaps longer than max_coast_frames start a new
                                track and identity is lost
    retroactive track start     M-of-N confirmation (a track outside the control
                                region appears min_track_observations frames
                                late).  Inside the control region the safety
                                pass-through emits it on frame one regardless,
                                so no obstacle that matters is ever delayed.
    full-track duplicate /      running per-frame evidence: co-location counters
    plausibility statistics     and observed-so-far medians, evaluated each frame
    ==========================  ==================================================

    Ego yaw rate may be supplied externally via ``ego_yaw_rate``; when it is,
    the internal bootstrap tracker is skipped entirely (roughly half the work).
    Otherwise it is estimated from this frame's velocity field and passed
    through a causal exponential filter -- no future frames, no RTS.
    """

    def __init__(self, cfg: Optional[CleanerConfig] = None):
        self.cfg = cfg if cfg is not None else CleanerConfig()
        boot_cfg = CleanerConfig(**{**asdict(self.cfg), "use_ego_motion": False})
        self._boot = Tracker(boot_cfg)
        self._main = Tracker(self.cfg)
        self._rs = np.random.RandomState(0)          # deterministic
        self._psi_filt = ScalarCVFilter(self.cfg.ego_sigma_psi_meas,
                                        self.cfg.ego_sigma_psidot, DT,
                                        rate_damping=self.cfg.ego_causal_rate_damping)
        self._v_filt = ScalarCVFilter(self.cfg.ego_sigma_v_meas,
                                      self.cfg.ego_sigma_a, DT,
                                      rate_damping=self.cfg.ego_causal_rate_damping)
        self.frame = -1
        self.ego_speed_mps = 0.0
        self.ego_yaw_rate = 0.0
        self._ego_seeded = False
        self._wy_hist: Dict[int, List[float]] = {}   # |w_y| at observed frames
        self._wx_hist: Dict[int, List[float]] = {}
        self._pair_run: Dict[Tuple[int, int], int] = {}
        self.n_restored = 0

    # -- ego motion ------------------------------------------------------
    def _update_ego(self, dets: Sequence[Detection], dt: float,
                    ego_yaw_rate: Optional[float],
                    ego_speed_mps: Optional[float]) -> float:
        cfg = self.cfg
        if ego_yaw_rate is not None:
            self.ego_yaw_rate = float(ego_yaw_rate)
            if ego_speed_mps is not None:
                self.ego_speed_mps = float(ego_speed_mps)
            return self.ego_yaw_rate
        if not cfg.use_ego_motion:
            return 0.0
        # Bootstrap tracker runs with zero ego input, so its (wx, wy) is the raw
        # ego-relative velocity field the static-point model needs.
        self._boot.step(self.frame, dets, dt)
        pts = ego_points_from_tracks(self._boot.active, self.frame, cfg, causal=True)
        r = solve_ego_frame(pts, cfg, self._rs)
        z_v = z_psi = None
        if r is not None and r[2] >= cfg.ego_min_inliers:
            z_v, z_psi = r[0], r[1]
        self.ego_speed_mps = self._v_filt.step(z_v)
        self.ego_yaw_rate = self._psi_filt.step(z_psi)
        self.ego_speed_mps = float(np.clip(self.ego_speed_mps, 0.0, cfg.ego_v_max))
        self.ego_yaw_rate = float(np.clip(self.ego_yaw_rate,
                                          -cfg.ego_psi_max, cfg.ego_psi_max))
        return self.ego_yaw_rate

    # -- running track statistics (the causal stand-in for whole-track ones)
    def _update_track_stats(self) -> None:
        cfg = self.cfg
        for t in self._main.active:
            if t.last_obs_frame == self.frame:
                self._wy_hist.setdefault(t.tid, []).append(abs(float(t.state[3])))
                self._wx_hist.setdefault(t.tid, []).append(abs(float(t.state[2])))
            wy = self._wy_hist.get(t.tid)
            if wy:
                cap = cfg.max_lateral_world_speed_mps.get(t.group, 8.0)
                t.plausible = bool(float(np.median(wy)) <= cap
                                   and float(np.median(self._wx_hist[t.tid]))
                                   <= cfg.max_radial_world_speed_mps)
            t.is_static = (t.group in cfg.static_prior_groups)

    def _update_duplicates(self) -> None:
        """Accumulate co-location evidence and flag redundant tracks.

        Same rule as the offline pass, but the evidence is a RUNNING counter of
        consecutive frames two tracks have stayed close, so the decision uses
        only what has already happened.  Once flagged, a track stays flagged.
        """
        cfg = self.cfg
        live = set()
        act = [t for t in self._main.active if t.duplicate_of is None]
        for i, a in enumerate(act):
            for b in act[i + 1:]:
                if a.group != b.group:
                    continue
                rad = cfg.duplicate_radius_m.get(a.group, 3.0)
                if math.hypot(a.state[0] - b.state[0], a.state[1] - b.state[1]) > rad:
                    continue
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                live.add(key)
                run = self._pair_run.get(key, 0) + 1
                self._pair_run[key] = run
                if run < cfg.duplicate_min_overlap_frames:
                    continue
                keep, drop = (a, b) if a.n_obs > b.n_obs else (b, a)
                if a.n_obs == b.n_obs:
                    # Tie: keep the nearer track, the conservative choice for
                    # the target-speed policy.
                    ra = math.hypot(a.state[0], a.state[1])
                    rb = math.hypot(b.state[0], b.state[1])
                    keep, drop = (a, b) if ra <= rb else (b, a)
                if (keep.n_obs >= cfg.duplicate_min_support
                        or keep.n_obs >= cfg.duplicate_support_ratio * drop.n_obs):
                    drop.duplicate_of = keep.tid
        for key in list(self._pair_run):
            if key not in live:
                del self._pair_run[key]

    # -- main entry point -------------------------------------------------
    def update(self, matrix: np.ndarray, dt: float = DT,
               ego_yaw_rate: Optional[float] = None,
               ego_speed_mps: Optional[float] = None
               ) -> Tuple[np.ndarray, List[dict]]:
        """Clean one frame.  Returns (cleaned 120x80 uint8, emitted objects).

        Parameters
        ----------
        matrix        : the raw 120x80 uint8 semantic matrix for this frame.
        dt            : seconds since the previous call (0.10 for this dataset).
        ego_yaw_rate  : measured ego yaw rate [rad/s], positive turning right.
                        When given, the internal estimator is bypassed.
        ego_speed_mps : measured ego speed, used only for the static/dynamic
                        annotation; it is deliberately NOT a model input (see
                        the module docstring).
        """
        m = np.asarray(matrix)
        if m.shape != (ROWS, COLS):
            raise ValueError(f"matrix must be {(ROWS, COLS)}, got {m.shape}")
        cfg = self.cfg
        self.frame += 1
        f = self.frame

        dets = extract_detections(m, cfg)
        psi = self._update_ego(dets, dt, ego_yaw_rate, ego_speed_mps)
        self._main.step(f, dets, dt, psi=psi)
        self._update_track_stats()
        self._update_duplicates()

        # ---- emission: established tracks, coasted up to the emit budget ----
        ems: List[dict] = []
        for t in self._main.active:
            established = (t.plausible and t.duplicate_of is None
                           and t.n_obs >= cfg.min_track_observations)
            if not established:
                continue
            coast = f - t.last_obs_frame
            if coast > cfg.emit_coast_frames:
                continue
            x, y = float(t.state[0]), float(t.state[1])
            row = EGO_ROW - int(round(x))
            col = EGO_COL + int(round(y))
            if not (0 <= row < ROWS and 0 <= col < COLS):
                continue
            ems.append(dict(tid=t.tid, cls=t.class_at(f), x=x, y=y,
                            wx=float(t.state[2]), wy=float(t.state[3]),
                            row=row, col=col, observed=(coast == 0),
                            static=t.is_static, group=t.group, n_obs=t.n_obs,
                            provenance=PROV_OBSERVED if coast == 0 else PROV_COASTED))

        wrapped = [ems]
        self.n_restored += add_safety_passthrough(wrapped, [dets], cfg)
        return rasterise(wrapped[0]), wrapped[0]

    def reset(self) -> None:
        """Drop all state; the next update starts a fresh sequence."""
        self.__init__(self.cfg)

# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    cleaned: np.ndarray                 # (F, 120, 80) uint8
    tracks: List[Track]
    emissions: List[List[dict]]
    ego_v: np.ndarray
    ego_psi: np.ndarray
    ego_inliers: np.ndarray
    mode: str
    config: CleanerConfig


def clean_sequence(matrices: Sequence[np.ndarray],
                   cfg: Optional[CleanerConfig] = None,
                   mode: str = "offline",
                   verbose: bool = True) -> CleanResult:
    """Clean a whole sequence.

    mode="offline"  : uses future frames (RTS smoothing, tracklet stitching,
                      duplicate merging, retro-active track start).  NOT
                      deployable as-is in real time.
    mode="causal"   : runs OnlineTemporalCleaner frame by frame.  This is the
                      SAME code a live pipeline calls, so the batch causal
                      result is bit-identical to streaming it -- there is only
                      one causal implementation, not a batch approximation of
                      one.
    """
    if cfg is None:
        cfg = CleanerConfig()
    if mode not in ("offline", "causal"):
        raise ValueError("mode must be 'offline' or 'causal'")

    n_frames = len(matrices)

    if mode == "causal":
        return _clean_causal(matrices, cfg, verbose)

    dets_per_frame = [extract_detections(m, cfg) for m in matrices]

    # --- pass 1: bootstrap tracker with no ego-motion input ----------------
    boot = Tracker(CleanerConfig(**{**asdict(cfg), "use_ego_motion": False}))
    for f in range(n_frames):
        boot.step(f, dets_per_frame[f])
    boot_tracks = boot.finish()
    if verbose:
        print(f"  [pass 1] bootstrap tracks: {len(boot_tracks)}")

    # --- pass 2: ego motion -------------------------------------------------
    if cfg.use_ego_motion:
        ego_v, ego_psi, ego_in = estimate_ego_motion(boot_tracks, n_frames, cfg)
    else:
        ego_v = np.zeros(n_frames); ego_psi = np.zeros(n_frames)
        ego_in = np.zeros(n_frames, dtype=int)
    if verbose:
        ok = ego_in >= cfg.ego_min_inliers
        print(f"  [pass 2] ego motion solved on {int(ok.sum())}/{n_frames} frames; "
              f"median v={np.median(ego_v[ok]) if ok.any() else float('nan'):.2f} m/s "
              f"({(np.median(ego_v[ok]) / 0.44704) if ok.any() else float('nan'):.1f} mph), "
              f"|psi| p95={np.percentile(np.abs(ego_psi), 95):.3f} rad/s")

    # --- pass 3: main tracker ----------------------------------------------
    trk = Tracker(cfg, ego_v, ego_psi)
    yaw_ok = trk.yaw_ok
    for f in range(n_frames):
        trk.step(f, dets_per_frame[f])
    tracks = trk.finish()
    if verbose:
        print(f"  [pass 3] tracks: {len(tracks)}")

    # --- pass 4: offline-only refinement ------------------------------------
    for t in tracks:
        _refilter(t, cfg, ego_psi, yaw_ok)
    tracks = stitch_tracklets(tracks, cfg, ego_psi, yaw_ok)
    tracks = merge_duplicate_tracks(tracks, cfg, ego_psi, yaw_ok)
    for t in tracks:
        _refilter(t, cfg, ego_psi, yaw_ok)
    flag_plausibility(tracks, cfg)
    suppress_duplicates(tracks, cfg)
    classify_static(tracks, cfg, ego_v)
    if verbose:
        print(f"  [pass 4] after stitch+merge: {len(tracks)} tracks "
              f"({sum(1 for t in tracks if t.is_static)} static, "
              f"{sum(1 for t in tracks if not t.plausible)} implausible, "
              f"{sum(1 for t in tracks if t.duplicate_of is not None)} duplicate)")

    emissions = build_emissions(tracks, n_frames, cfg, ego_psi,
                                causal=False, yaw_ok=yaw_ok)
    restored = add_safety_passthrough(emissions, dets_per_frame, cfg)
    if verbose:
        print(f"  [safety] restored {restored} control-relevant detections "
              f"the tracker had not represented")
    cleaned = np.stack([rasterise(emissions[f]) for f in range(n_frames)])

    return CleanResult(cleaned=cleaned, tracks=tracks, emissions=emissions,
                       ego_v=ego_v, ego_psi=ego_psi, ego_inliers=ego_in,
                       mode="offline", config=cfg)


def _clean_causal(matrices: Sequence[np.ndarray], cfg: CleanerConfig,
                  verbose: bool) -> CleanResult:
    """Batch wrapper that simply streams the frames through the online cleaner."""
    n_frames = len(matrices)
    online = OnlineTemporalCleaner(cfg)
    cleaned = np.zeros((n_frames, ROWS, COLS), dtype=np.uint8)
    emissions: List[List[dict]] = []
    ego_v = np.zeros(n_frames)
    ego_psi = np.zeros(n_frames)
    for f, m in enumerate(matrices):
        cleaned[f], ems = online.update(m, dt=DT)
        emissions.append(ems)
        ego_v[f] = online.ego_speed_mps
        ego_psi[f] = online.ego_yaw_rate
    tracks = online._main.finish()
    if verbose:
        print(f"  [causal] streamed {n_frames} frames; tracks: {len(tracks)}; "
              f"median v={np.median(ego_v):.2f} m/s "
              f"({np.median(ego_v) / 0.44704:.1f} mph), "
              f"|psi| p95={np.percentile(np.abs(ego_psi), 95):.3f} rad/s")
        print(f"  [safety] restored {online.n_restored} control-relevant "
              f"detections the tracker had not represented")
    return CleanResult(cleaned=cleaned, tracks=tracks, emissions=emissions,
                       ego_v=ego_v, ego_psi=ego_psi,
                       ego_inliers=np.zeros(n_frames, dtype=int),
                       mode="causal", config=cfg)
