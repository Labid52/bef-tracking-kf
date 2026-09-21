#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import time
import serial

import control_params
from target_speed import getTargetSpeed


ser = serial.Serial(port='/dev/ttyUSB0', baudrate=1000000, timeout=0.1, write_timeout=0.1)

# Allow time for the serial connection to initialize
time.sleep(2)

# Parameters and paths now come from control_params.py (single source of truth).
# Two things were corrected here: dt was 0.05 while this dataset is 10 FPS
# (dt = 0.10), and MATRIX_DIR pointed at a different checkout of the project.
MATRIX_DIR = control_params.MATRIX_DIR

WIDTH_OF_INTEREST_M = control_params.WIDTH_OF_INTEREST_M
EGO_SPEED_MPH = control_params.EGO_SPEED_MPH

SAFETY_STOPPING_DISTANCE_M = control_params.SAFETY_STOPPING_DISTANCE_M
STOP_OFFSET_M = control_params.STOP_OFFSET_M

ACCELERATION_MPS2 = control_params.ACCELERATION_MPS2
DECELERATION_MPS2 = control_params.DECELERATION_MPS2
DT = control_params.DT

MAX_TARGET_SPEED_MPH = control_params.MAX_TARGET_SPEED_MPH
HOOD_IGNORE_DISTANCE_M = control_params.HOOD_IGNORE_DISTANCE_M


def main():
    files = sorted(MATRIX_DIR.glob("*.npy"))

    print(f"{'frame':<14} {'target_speed_mph':>18}")
    print("-" * 34)

    for path in files:
        matrix = np.load(path)

        target_speed_mph = getTargetSpeed(
            matrix=matrix,
            width_of_interest_m=WIDTH_OF_INTEREST_M,
            ego_speed_mph=EGO_SPEED_MPH,
            safety_stopping_distance_m=SAFETY_STOPPING_DISTANCE_M,
            acceleration_mps2=ACCELERATION_MPS2,
            deceleration_mps2=DECELERATION_MPS2,
            dt=DT,
            max_target_speed_mph=MAX_TARGET_SPEED_MPH,
            hood_ignore_distance_m=HOOD_IGNORE_DISTANCE_M,
            stop_offset_m=STOP_OFFSET_M,
        )

        print(f"{path.name:<14} {target_speed_mph:>18.2f}")
        ser.write(f"V:{target_speed_mph:.2f}\n".encode("ascii"))

        # # Uncomment them if you want to make sure the ESP32 received the message.   
        # response = ser.readline().decode("ascii", errors="replace").strip()
        # if response:
        #     print("ESP32:", response)

if __name__ == "__main__":
    main()
