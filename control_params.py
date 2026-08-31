#!/usr/bin/env python3
"""
Single source of truth for dataset timing and longitudinal-control parameters.

Every runner, validator and animation imports from here so the timing can no
longer drift between scripts.  Before this module existed, four scripts each
carried their own copy of the getTargetSpeed keyword set and passed dt = 0.05
while the dataset is 10 FPS (dt = 0.10).

Nothing here redefines control policy: the policy lives in target_speed.py and
is called unmodified.  These are the ARGUMENTS the policy is invoked with.

Note that ``dt`` inside TARGET_SPEED_KW is only carried through to
getTargetSpeed for future closed-loop use; the present policy is distance-only,
so it does not change the returned speed.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Timing.  There are two distinct rates in this project; keep them separate.
#
#   LIVE_FPS    the nominal rate of the deployed BEV pipeline (20 Hz).  The
#               live tracker still receives the ACTUAL measured dt on every
#               update -- this constant is only the nominal default.
#   LEGACY_FPS  the rate of the recorded datasets stored in this repository
#               (matrix/ and realtime_capture/matrix), which were captured at
#               10 Hz.  Reprocess those with --dt 0.10.
#
# Nothing in the tracker reads a fixed rate: every physical computation takes dt
# as an explicit argument, so the same code runs at either rate.
# ---------------------------------------------------------------------------
LIVE_FPS = 20.0
LIVE_DT = 1.0 / LIVE_FPS        # 0.05 s -- nominal live deployment
LEGACY_FPS = 10.0
LEGACY_DT = 1.0 / LEGACY_FPS    # 0.10 s -- recorded datasets in this repo

# Default used by the recorded-data tools in this repo (the datasets are 10 Hz).
# Live code must pass its own measured dt instead of relying on this.
FPS = LEGACY_FPS
DT = LEGACY_DT
FRAME_STEP = 1          # every matrix frame is used exactly once

PROJECT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = PROJECT_DIR / "matrix"

# Longitudinal-control parameters passed to getTargetSpeed().
WIDTH_OF_INTEREST_M = 4.0
EGO_SPEED_MPH = 20.0
SAFETY_STOPPING_DISTANCE_M = 8.0
STOP_OFFSET_M = 5.0
ACCELERATION_MPS2 = 1.1176
DECELERATION_MPS2 = 1.1176
MAX_TARGET_SPEED_MPH = 20.0
HOOD_IGNORE_DISTANCE_M = 6.0
STOP_LATERAL_LIMIT_M = 15.0

#: Keyword set for target_speed.getTargetSpeed(matrix=..., **TARGET_SPEED_KW)
TARGET_SPEED_KW = dict(
    width_of_interest_m=WIDTH_OF_INTEREST_M,
    ego_speed_mph=EGO_SPEED_MPH,
    safety_stopping_distance_m=SAFETY_STOPPING_DISTANCE_M,
    acceleration_mps2=ACCELERATION_MPS2,
    deceleration_mps2=DECELERATION_MPS2,
    dt=DT,
    max_target_speed_mph=MAX_TARGET_SPEED_MPH,
    hood_ignore_distance_m=HOOD_IGNORE_DISTANCE_M,
    stop_offset_m=STOP_OFFSET_M,
    stop_lateral_limit_m=STOP_LATERAL_LIMIT_M,
)
