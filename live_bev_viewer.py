#!/usr/bin/env python3
"""
Live / causal BEV viewer for the longitudinal-control project.

This file DOES NOT change the accepted tracking method.  It is only a live
wrapper around ``OnlineTemporalCleaner.update()``.

Production use
--------------
Create ONE ``LiveBEVProcessor`` when the perception process starts.  For every
new 120x80 semantic matrix, call ``processor.update(raw_matrix, ...)`` exactly
once.  The processor keeps all Kalman / track history internally and returns
the stabilized matrix and current tracked-object states immediately.

Recorded-data replay
--------------------
This script can also replay the existing ``matrix/*.npy`` files at 10 Hz.  The
files are consumed one at a time in chronological order; future matrices are
never given to the cleaner.  This is useful for testing the exact live code
path before connecting it to the NVIDIA DRIVE Thor perception publisher.

Examples
--------
    python3 live_bev_viewer.py
    python3 live_bev_viewer.py --start 1110 --end 1190
    python3 live_bev_viewer.py --no-realtime
    python3 live_bev_viewer.py --headless --no-realtime

Keys in the OpenCV window
-------------------------
    q / ESC : quit
    p       : pause / resume replay
    r       : reset the causal tracker
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import control_params as cp
import temporal_matrix_cleaner as tmc
from temporal_matrix_cleaner import OnlineTemporalCleaner
from target_speed import getTargetSpeed
from longitudinal_safety import (
    BEVLongitudinalAdapterConfig,
    LongitudinalSafetyConfig,
    LongitudinalSafetyResult,
    evaluate_longitudinal_safety,
)


# ---------------------------------------------------------------------------
# Visualization metadata only.  These values do not affect tracking/control.
# OpenCV colors are BGR.
# ---------------------------------------------------------------------------
CLASS_STYLE = {
    1: ("person",        (60, 70, 230)),
    2: ("bicycle",       (60, 135, 230)),
    3: ("car",           (230, 140, 60)),
    4: ("motorcycle",    (220, 80, 150)),
    5: ("bus",           (180, 160, 20)),
    6: ("truck",         (160, 95, 30)),
    7: ("stop sign",     (20, 20, 220)),
    8: ("traffic light", (135, 135, 135)),
    9: ("red light",     (20, 20, 220)),
    10: ("yellow light", (20, 190, 220)),
    11: ("green light",  (40, 180, 40)),
}
DEFAULT_STYLE = ("unknown", (150, 150, 150))


@dataclass
class LiveFrameResult:
    """Output of one real-time cleaner update."""

    frame_index: int
    cleaned_matrix: np.ndarray
    objects: List[dict]
    raw_target_speed_mph: float
    cleaned_target_speed_mph: float
    tracker_latency_ms: float
    active_tracks: int
    coasted_tracks: int
    # Backwards-compatible alias for the tracker-internal estimate.  New code
    # should use the two explicit fields below and never infer safety validity
    # from this legacy name.
    ego_speed_mps: float
    safety_input_ego_speed_mps: Optional[float]
    tracker_ego_speed_mps: float
    ego_yaw_rate_rps: float
    nominal_target_speed_mph: float
    safe_target_speed_mph: float
    safety_zone: str
    critical_boundary_m: Optional[float]
    collision_boundary_m: Optional[float]
    obstacle_distance_m: Optional[float]
    safety_reason: str
    safety_result: Optional[LongitudinalSafetyResult]
    obstacle_gap_description: str
    dt_s: float = cp.LIVE_DT
    elapsed_s: float = 0.0


class LiveBEVProcessor:
    """Persistent causal perception post-processor.

    Instantiate this class ONCE.  Each call to :meth:`update` consumes only the
    current matrix; all past information needed by the Kalman/data-association
    tracker is retained inside ``self.cleaner``.
    """

    def __init__(
        self,
        nominal_dt: float = cp.LIVE_DT,
        safety_config: Optional[LongitudinalSafetyConfig] = None,
        bev_adapter_config: Optional[BEVLongitudinalAdapterConfig] = None,
    ) -> None:
        self.cleaner = OnlineTemporalCleaner()
        self.safety_config = safety_config or LongitudinalSafetyConfig()
        self.safety_config.validate()
        self.bev_adapter_config = bev_adapter_config or BEVLongitudinalAdapterConfig()
        self.bev_adapter_config.validate()
        self.frame_index = -1
        self.nominal_dt = float(nominal_dt)
        self.elapsed_s = 0.0
        self._last_monotonic: Optional[float] = None
        self.dt_clamped = 0          # times a measured dt hit the guard rails

    def measure_dt(self) -> float:
        """Interval since the previous call, from the monotonic clock.

        Use this when the caller has no timestamp of its own::

            cleaned, objs = proc.update(matrix, dt=proc.measure_dt())

        The first call returns the nominal dt because there is no predecessor.
        """
        now = time.monotonic()
        if self._last_monotonic is None:
            self._last_monotonic = now
            return self.nominal_dt
        dt = now - self._last_monotonic
        self._last_monotonic = now
        return dt

    def update(
        self,
        raw_matrix: np.ndarray,
        *,
        dt: Optional[float] = None,
        ego_yaw_rate: Optional[float] = None,
        ego_speed_mps: Optional[float] = None,
    ) -> LiveFrameResult:
        """Process exactly one current 120x80 semantic matrix.

        Parameters
        ----------
        raw_matrix:
            Current semantic matrix.  No previous/future matrices are required.
        dt:
            Measured seconds since the previous update.  Pass the real interval;
            the tracker is fully rate-aware.  ``None`` uses the nominal rate
            (20 Hz live, or whatever this processor was constructed with).

            ABNORMAL dt POLICY: a value outside
            [temporal_matrix_cleaner.DT_MIN, DT_MAX] = [0.005, 0.5] s is clamped
            into that range and counted in ``self.dt_clamped``.  Clamping is
            deliberate: a scheduling stall of several seconds must not be turned
            into several seconds of blind constant-velocity extrapolation.  The
            coast policy then ages the affected tracks out normally.
        ego_yaw_rate:
            Optional measured yaw rate [rad/s].  If available from the vehicle
            IMU/localization stack, pass it here; otherwise the existing causal
            scene-based estimator is used.
        ego_speed_mps:
            Explicit safety input [m/s].  The same value is also forwarded to
            the existing cleaner for annotation, but it remains distinct from
            the cleaner's internally maintained ego-speed estimate.
        """
        raw = np.asarray(raw_matrix)
        if raw.shape != (tmc.ROWS, tmc.COLS):
            raise ValueError(
                f"raw_matrix must have shape {(tmc.ROWS, tmc.COLS)}, got {raw.shape}"
            )
        if raw.dtype != np.uint8:
            raw = raw.astype(np.uint8, copy=False)

        if ego_speed_mps is not None:
            ego_speed_mps = float(ego_speed_mps)
            if not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
                raise ValueError("ego_speed_mps must be finite and non-negative")

        if dt is None:
            dt = self.nominal_dt
        dt = float(dt)
        if not (tmc.DT_MIN <= dt <= tmc.DT_MAX):
            self.dt_clamped += 1
            dt = min(max(dt, tmc.DT_MIN), tmc.DT_MAX)

        self.frame_index += 1
        self.elapsed_s += dt

        t0 = time.perf_counter()
        cleaned, objects = self.cleaner.update(
            raw,
            dt=dt,
            ego_yaw_rate=ego_yaw_rate,
            ego_speed_mps=ego_speed_mps,
        )
        tracker_latency_ms = (time.perf_counter() - t0) * 1e3

        ts_raw = float(getTargetSpeed(matrix=raw, **cp.TARGET_SPEED_KW))
        ts_clean = float(getTargetSpeed(matrix=cleaned, **cp.TARGET_SPEED_KW))
        safety_result: Optional[LongitudinalSafetyResult] = None
        if ego_speed_mps is not None:
            safety_result = evaluate_longitudinal_safety(
                matrix=cleaned,
                ego_speed_mps=ego_speed_mps,
                nominal_target_speed_mph=ts_clean,
                dt=dt,
                config=self.safety_config,
                bev_config=self.bev_adapter_config,
            )

        coasted = sum(
            1 for obj in objects
            if obj.get("provenance") == tmc.PROV_COASTED or not bool(obj.get("observed", True))
        )

        return LiveFrameResult(
            frame_index=self.frame_index,
            cleaned_matrix=cleaned,
            objects=objects,
            raw_target_speed_mph=ts_raw,
            cleaned_target_speed_mph=ts_clean,
            tracker_latency_ms=tracker_latency_ms,
            active_tracks=len(objects),
            coasted_tracks=coasted,
            # Preserve the old field's tracker-estimate meaning for existing
            # consumers; safety diagnostics use safety_input_ego_speed_mps.
            ego_speed_mps=float(self.cleaner.ego_speed_mps),
            safety_input_ego_speed_mps=ego_speed_mps,
            tracker_ego_speed_mps=float(self.cleaner.ego_speed_mps),
            ego_yaw_rate_rps=float(self.cleaner.ego_yaw_rate),
            nominal_target_speed_mph=ts_clean,
            safe_target_speed_mph=(safety_result.safe_target_speed_mph
                                   if safety_result is not None else ts_clean),
            safety_zone=(safety_result.zone if safety_result is not None else "UNAVAILABLE"),
            critical_boundary_m=(safety_result.critical_boundary_m
                                 if safety_result is not None else None),
            collision_boundary_m=(safety_result.collision_boundary_m
                                  if safety_result is not None else None),
            obstacle_distance_m=(safety_result.obstacle_distance_m
                                 if safety_result is not None else None),
            safety_reason=(safety_result.reason if safety_result is not None else
                           "three-zone safety requires explicit ego_speed_mps"),
            safety_result=safety_result,
            obstacle_gap_description=(
                "front-bumper gap"
                if self.bev_adapter_config.bev_origin_to_front_bumper_m is not None
                else "provisional BEV range"
            ),
            dt_s=dt,
            elapsed_s=self.elapsed_s,
        )

    def reset(self) -> None:
        """Clear all temporal state (for a new drive/route/session)."""
        self.cleaner.reset()
        self.frame_index = -1
        self.elapsed_s = 0.0
        self._last_monotonic = None
        self.dt_clamped = 0


class BEVRenderer:
    """OpenCV renderer for raw vs. live causal BEV.

    Rendering is intentionally separate from :class:`LiveBEVProcessor`, so the
    control path can use ``cleaned_matrix`` even when visualization is disabled.
    """

    def __init__(
        self,
        pixels_per_meter: float = 6.0,
        x_min_m: float = -20.0,
        x_max_m: float = 80.0,
        y_min_m: float = -40.0,
        y_max_m: float = 40.0,
    ) -> None:
        self.ppm = float(pixels_per_meter)
        self.x_min = float(x_min_m)
        self.x_max = float(x_max_m)
        self.y_min = float(y_min_m)
        self.y_max = float(y_max_m)
        self.panel_w = int(round((self.y_max - self.y_min) * self.ppm))
        self.panel_h = int(round((self.x_max - self.x_min) * self.ppm))
        self.info_h = 150

        # ---- VISUALIZATION-ONLY state ---------------------------------------
        # Previous displayed continuous position per track id, used solely to
        # draw the motion arrow along the apparent BEV motion the viewer sees.
        # This is never read by OnlineTemporalCleaner, the Kalman state, the
        # association, the target-speed computation or the cleaned matrix.
        #   previous_display_position[tid] = (x_forward_m, y_right_m, t_seconds)
        self.previous_display_position: dict = {}
        self._display_clock_s = 0.0
        # Drop visualization history this long after a track stops being drawn,
        # so a later track id can never inherit a stale arrow state.
        self._display_history_ttl_s = 1.0

    def _xy_to_px(self, x_forward: float, y_right: float) -> Tuple[int, int]:
        px = int(round((y_right - self.y_min) * self.ppm))
        py = int(round((self.x_max - x_forward) * self.ppm))
        return px, py

    def _base_panel(self, title: str, subtitle: str = "") -> np.ndarray:
        img = np.full((self.panel_h, self.panel_w, 3), 248, dtype=np.uint8)

        # Grid every 10 m.
        for x in np.arange(np.ceil(self.x_min / 10) * 10, self.x_max + 0.1, 10):
            p1 = self._xy_to_px(float(x), self.y_min)
            p2 = self._xy_to_px(float(x), self.y_max)
            cv2.line(img, p1, p2, (220, 220, 220), 1, cv2.LINE_AA)
        for y in np.arange(np.ceil(self.y_min / 10) * 10, self.y_max + 0.1, 10):
            p1 = self._xy_to_px(self.x_min, float(y))
            p2 = self._xy_to_px(self.x_max, float(y))
            cv2.line(img, p1, p2, (225, 225, 225), 1, cv2.LINE_AA)

        # Object corridor used by target speed: +/- width_of_interest/2.
        hw = cp.WIDTH_OF_INTEREST_M / 2.0
        left_top = self._xy_to_px(self.x_max, -hw)
        right_bottom = self._xy_to_px(0.0, hw)
        overlay = img.copy()
        cv2.rectangle(overlay, left_top, right_bottom, (245, 230, 215), -1)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

        # Stop-sign lateral gate +/- 15 m, outline only.
        stop_left_top = self._xy_to_px(self.x_max, -cp.STOP_LATERAL_LIMIT_M)
        stop_right_bottom = self._xy_to_px(0.0, cp.STOP_LATERAL_LIMIT_M)
        cv2.rectangle(img, stop_left_top, stop_right_bottom, (190, 190, 240), 1)

        # Ego axes and marker.
        p0 = self._xy_to_px(0.0, 0.0)
        cv2.line(img, self._xy_to_px(self.x_min, 0), self._xy_to_px(self.x_max, 0),
                 (100, 100, 100), 1, cv2.LINE_AA)
        cv2.circle(img, p0, 7, (20, 20, 20), -1, cv2.LINE_AA)
        tip = self._xy_to_px(3.0, 0.0)
        cv2.arrowedLine(img, p0, tip, (20, 20, 20), 2, tipLength=0.35)

        cv2.putText(img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (20, 20, 20), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (70, 70, 70), 1, cv2.LINE_AA)
        return img

    def reset_display_history(self) -> None:
        """Forget the visualization-only arrow history.

        Called whenever the tracker is reset, because track ids restart from
        zero and a new track must not inherit the previous session's position.
        """
        self.previous_display_position.clear()
        self._display_clock_s = 0.0

    @staticmethod
    def _style(cls: int) -> Tuple[str, Tuple[int, int, int]]:
        return CLASS_STYLE.get(int(cls), DEFAULT_STYLE)

    def _draw_raw(self, raw: np.ndarray) -> np.ndarray:
        img = self._base_panel("RAW CURRENT-FRAME DETECTIONS", "current matrix only")
        rr, cc = np.nonzero(raw)
        for r, c in zip(rr.tolist(), cc.tolist()):
            cls = int(raw[r, c])
            x = float(tmc.EGO_ROW - r)
            y = float(c - tmc.EGO_COL)
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                continue
            name, color = self._style(cls)
            px = self._xy_to_px(x, y)
            cv2.circle(img, px, 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, px, 6, (30, 30, 30), 1, cv2.LINE_AA)
        return img

    def _draw_tracks(self, objects: List[dict], dt: float = cp.DT) -> np.ndarray:
        img = self._base_panel("ONLINE / CAUSAL TRACKING", "Future frames used: NO")
        # Advance the visualization clock by the real elapsed time of this
        # update, so the arrow uses the ACTUAL interval between the two
        # displayed states even if a track was not drawn for a few frames.
        self._display_clock_s += float(dt)
        now = self._display_clock_s
        seen_this_frame = set()
        for obj in objects:
            x = float(obj["x"])
            y = float(obj["y"])
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                continue
            cls = int(obj["cls"])
            tid = int(obj.get("tid", -1))
            name, color = self._style(cls)
            px = self._xy_to_px(x, y)

            cv2.circle(img, px, 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, px, 7, (30, 30, 30), 1, cv2.LINE_AA)

            prov = obj.get("provenance", tmc.PROV_OBSERVED)
            if prov == tmc.PROV_COASTED:
                cv2.circle(img, px, 10, (0, 140, 255), 2, cv2.LINE_AA)
                status = " C"
            elif prov == tmc.PROV_PASSTHROUGH:
                cv2.circle(img, px, 10, (30, 180, 30), 2, cv2.LINE_AA)
                status = " S"
            else:
                status = ""

            label = f"{name} #{tid}{status}" if tid >= 0 else f"{name}{status}"
            cv2.putText(img, label, (px[0] + 8, px[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1,
                        cv2.LINE_AA)

            # ---- Velocity arrow: visualization only, never fed back ----------
            # It must point along the motion the viewer actually sees, so it is
            # derived from THIS track's previous and current displayed
            # continuous positions rather than from the tracker's internal
            # Kalman state (wx, wy), which is yaw-compensated and therefore not
            # always the apparent on-screen motion:
            #
            #     vx_display = (x_curr - x_prev) / dt_display
            #     vy_display = (y_curr - y_prev) / dt_display
            #
            # Pass-through emissions share tid = -1 and are not one persistent
            # object, so they are never chained.  A track with only one displayed
            # position yet gets no arrow.
            if tid >= 0:
                seen_this_frame.add(tid)
                prev = self.previous_display_position.get(tid)
                if prev is not None:
                    x_prev, y_prev, t_prev = prev
                    dt_display = now - t_prev
                    if dt_display > 0.0:
                        vx = (x - x_prev) / dt_display
                        vy = (y - y_prev) / dt_display
                        mag = float(np.hypot(vx, vy))
                        if mag > 0.25:
                            horizon_s = min(0.5, 4.0 / mag)
                            qx = x + vx * horizon_s
                            qy = y + vy * horizon_s
                            q = self._xy_to_px(qx, qy)
                            cv2.arrowedLine(img, px, q, (70, 70, 70), 1,
                                            cv2.LINE_AA, tipLength=0.25)
                self.previous_display_position[tid] = (x, y, now)

        # Expire visualization history for tracks that are no longer drawn.
        for tid_old in [t for t, v in self.previous_display_position.items()
                        if t not in seen_this_frame
                        and now - v[2] > self._display_history_ttl_s]:
            del self.previous_display_position[tid_old]
        return img

    def render(self, raw: np.ndarray, result: LiveFrameResult,
               dt: float = cp.DT) -> np.ndarray:
        """``dt`` is the real interval since the previous rendered frame; it is
        used only for the visualization arrow direction."""
        left = self._draw_raw(raw)
        right = self._draw_tracks(result.objects, dt=dt)
        body = np.hstack([left, right])

        info = np.full((self.info_h, body.shape[1], 3), 250, dtype=np.uint8)
        elapsed = getattr(result, "elapsed_s", result.frame_index * cp.DT)
        line1 = (
            f"frame {result.frame_index:06d}   time {elapsed:7.1f} s   "
            f"tracker {result.tracker_latency_ms:6.2f} ms   "
            f"active {result.active_tracks:2d}   coasted {result.coasted_tracks:2d}"
        )
        if result.safety_result is None:
            line2 = (
                "Zone: UNAVAILABLE   Safety ego speed: not supplied   "
                "Obstacle distance: not evaluated"
            )
            line3 = (
                f"Critical boundary: n/a   Collision boundary: n/a   "
                f"Nominal target: {result.nominal_target_speed_mph:5.2f} mph   "
                f"Safe target: {result.safe_target_speed_mph:5.2f} mph (pass-through)"
            )
        else:
            ego_mph = result.safety_input_ego_speed_mps / 0.44704
            if result.obstacle_distance_m is None:
                horizon_m = result.safety_result.observation_horizon_m
                observed_text = (
                    "No obstacle observed within unbounded synthetic horizon"
                    if horizon_m is None else
                    f"No obstacle observed within {horizon_m:.1f} m provisional BEV horizon"
                )
            else:
                observed_text = (
                    f"{result.obstacle_gap_description}: "
                    f"{result.obstacle_distance_m:.1f} m"
                )
            line2 = (
                f"Zone: {result.safety_zone}   Safety ego speed: {ego_mph:5.1f} mph   "
                f"{observed_text}"
            )
            line3 = (
                f"Critical boundary: {result.critical_boundary_m:.1f} m   "
                f"Collision boundary: {result.collision_boundary_m:.1f} m   "
                f"Nominal target: {result.nominal_target_speed_mph:5.2f} mph   "
                f"Safe target: {result.safe_target_speed_mph:5.2f} mph"
            )
        line4 = (
            f"raw target {result.raw_target_speed_mph:5.2f} mph   "
            f"tracker ego estimate {result.tracker_ego_speed_mps:5.2f} m/s   "
            f"ego yaw {result.ego_yaw_rate_rps:+.3f} rad/s   "
            "q/ESC quit   p pause   r reset   C=coasted   S=passthrough"
        )
        if result.safety_result is None:
            line5 = "Coverage: n/a   Critical horizon max speed: n/a"
        else:
            max_critical_mps = result.safety_result.maximum_critical_horizon_speed_mps
            max_speed_text = (
                "n/a" if max_critical_mps is None
                else "unbounded" if math.isinf(max_critical_mps)
                else f"{max_critical_mps / 0.44704:.1f} mph"
            )
            line5 = (
                f"Coverage: {result.safety_result.coverage_status.replace('_', ' ')}   "
                f"Critical horizon max speed: {max_speed_text}"
            )
        cv2.putText(info, line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(info, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(info, line3, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(info, line4, (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(info, line5, (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (80, 80, 80), 1, cv2.LINE_AA)
        return np.vstack([body, info])


# ---------------------------------------------------------------------------
# Recorded sequence replay.  This exercises the SAME one-matrix update() API
# that the live perception callback will use on Thor.
# ---------------------------------------------------------------------------
def replay_matrix_directory(args: argparse.Namespace) -> None:
    files = sorted(Path(args.matrix_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy matrices found in {args.matrix_dir}")

    start = max(0, int(args.start))
    end = len(files) if args.end is None else min(len(files), int(args.end))
    files = files[start:end]
    if not files:
        raise ValueError("empty replay frame range")

    dt_nominal = float(args.dt)
    processor = LiveBEVProcessor(nominal_dt=dt_nominal)
    renderer = BEVRenderer(pixels_per_meter=args.ppm)
    paused = False
    latencies: List[float] = []

    print(f"Live causal replay: {len(files)} frames from {args.matrix_dir}")
    print(f"Timing: {1.0/dt_nominal:.1f} Hz, nominal dt={dt_nominal:.3f} s")
    print("Future frames are not supplied to OnlineTemporalCleaner.update().")

    try:
        for path in files:
            loop_start = time.perf_counter()
            raw = np.load(path)
            # ONE update per newly received matrix.  Replay uses the dataset's
            # nominal dt so results are reproducible; a real sensor callback
            # should pass processor.measure_dt() instead.
            result = processor.update(
                raw,
                dt=dt_nominal,
                ego_speed_mps=args.ego_speed_mps,
            )
            latencies.append(result.tracker_latency_ms)

            if not args.headless:
                canvas = renderer.render(raw, result, dt=dt_nominal)
                cv2.imshow(args.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    processor.reset()
                    renderer.reset_display_history()
                    print("tracker reset")
                if key == ord("p"):
                    paused = not paused

                while paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        return
                    if key == ord("p"):
                        paused = False
                    elif key == ord("r"):
                        processor.reset()
                        renderer.reset_display_history()
                        print("tracker reset")

            if args.realtime:
                remaining = dt_nominal - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    if latencies:
        a = np.asarray(latencies, dtype=float)
        print(
            "tracker update latency [ms]: "
            f"mean={a.mean():.2f}, p95={np.percentile(a,95):.2f}, "
            f"p99={np.percentile(a,99):.2f}, max={a.max():.2f}; "
            f"budget={dt_nominal*1000:.0f} ms "
            f"({1.0/dt_nominal:.0f} Hz)"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, default=cp.MATRIX_DIR,
                    help="recorded matrices for live-style replay")
    ap.add_argument("--dt", type=float, default=cp.LEGACY_DT,
                    help="nominal seconds per frame of the replayed sequence "
                         "(0.10 for the 10-Hz recordings in this repo, 0.05 to "
                         "exercise the 20-Hz live configuration)")
    ap.add_argument("--fps", type=float, default=None,
                    help="alternative to --dt (dt = 1/fps)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--ppm", type=float, default=6.0,
                    help="display pixels per meter (visualization only)")
    ap.add_argument(
        "--ego-speed-mps",
        type=float,
        default=None,
        help="explicit TEST/measured ego speed for three-zone safety; omitted "
             "means safety UNAVAILABLE and nominal target pass-through",
    )
    ap.add_argument("--window-name", default="Live Causal BEV")
    ap.add_argument("--headless", action="store_true",
                    help="run tracker without opening an OpenCV window")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="replay as fast as possible instead of sleeping to 10 Hz")
    ap.set_defaults(realtime=True)
    args = ap.parse_args()
    if args.fps is not None:
        args.dt = 1.0 / args.fps
    return args


def main() -> None:
    replay_matrix_directory(parse_args())


if __name__ == "__main__":
    main()
