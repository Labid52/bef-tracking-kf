import math
import numpy as np

EGO_ROW = 80
EGO_COL = 40
CELL_SIZE_M = 1.0

OBSTACLE_CLASSES = [1, 2, 3, 4, 5, 6]   # person, bicycle, car, motorcycle, bus, truck
STOP_CLASSES = [7, 9]                   # stop sign, red light

CAR_CLASS_ID = 3
MPS_TO_MPH = 1.0 / 0.44704


def getTargetSpeed(
    matrix,
    width_of_interest_m,
    ego_speed_mph,
    safety_stopping_distance_m,
    acceleration_mps2,
    deceleration_mps2,
    dt,
    max_target_speed_mph=20.0,
    hood_ignore_distance_m=6.0,
    stop_offset_m=5.0,
    stop_lateral_limit_m=15.0,
):
    """
    Return one target speed in mph from one 120x80 semantic matrix.

    Current logic:
    - Physical obstacles (classes 1-6) use width_of_interest_m
    - Stop sign and red light (classes 7 and 9) use stop_lateral_limit_m
    - Hood rejection applies only to cars
    - ego_speed_mph, acceleration_mps2, and dt are kept for future closed-loop use
    """

    matrix = np.asarray(matrix)

    if matrix.shape != (120, 80):
        raise ValueError("Matrix must have shape (120, 80)")

    half_width_m = width_of_interest_m / 2.0
    target_speed_mph = float(max_target_speed_mph)

    # =========================================================
    # 1) Physical obstacles: person, bicycle, car, motorcycle, bus, truck
    #    These must be inside the driving corridor.
    # =========================================================
    obstacle_rows, obstacle_cols = np.where(np.isin(matrix, OBSTACLE_CLASSES))

    if len(obstacle_rows) > 0:
        obstacle_classes = matrix[obstacle_rows, obstacle_cols]

        forward_distance_m = (EGO_ROW - obstacle_rows) * CELL_SIZE_M
        lateral_distance_m = (obstacle_cols - EGO_COL) * CELL_SIZE_M

        valid_obstacle = (
            (forward_distance_m > 0.0)
            & (np.abs(lateral_distance_m) <= half_width_m)
        )

        # Ignore hood only when the detected class is car
        hood_car = (
            (obstacle_classes == CAR_CLASS_ID)
            & (forward_distance_m <= hood_ignore_distance_m)
        )

        valid_obstacle = valid_obstacle & (~hood_car)

        if np.any(valid_obstacle):
            nearest_obstacle_distance_m = float(
                np.min(forward_distance_m[valid_obstacle])
            )

            usable_distance_m = max(
                0.0,
                nearest_obstacle_distance_m - safety_stopping_distance_m
            )

            obstacle_speed_mps = math.sqrt(
                2.0 * deceleration_mps2 * usable_distance_m
            )

            obstacle_speed_mph = obstacle_speed_mps * MPS_TO_MPH

            target_speed_mph = min(target_speed_mph, obstacle_speed_mph)

    # =========================================================
    # 2) Stop-required detections: stop sign and red light
    #    These can be at the side, so use a separate wider lateral limit.
    # =========================================================
    stop_rows, stop_cols = np.where(np.isin(matrix, STOP_CLASSES))

    if len(stop_rows) > 0:
        stop_forward_distance_m = (EGO_ROW - stop_rows) * CELL_SIZE_M
        stop_lateral_distance_m = (stop_cols - EGO_COL) * CELL_SIZE_M

        valid_stop = (
            (stop_forward_distance_m > 0.0)
            & (np.abs(stop_lateral_distance_m) <= stop_lateral_limit_m)
        )

        if np.any(valid_stop):
            nearest_stop_distance_m = float(
                np.min(stop_forward_distance_m[valid_stop])
            )

            available_stop_distance_m = max(
                0.0,
                nearest_stop_distance_m - stop_offset_m
            )

            stop_speed_mps = math.sqrt(
                2.0 * deceleration_mps2 * available_stop_distance_m
            )

            stop_speed_mph = stop_speed_mps * MPS_TO_MPH

            target_speed_mph = min(target_speed_mph, stop_speed_mph)

    return float(target_speed_mph)
