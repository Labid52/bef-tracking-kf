"""Full-matrix renderer for standalone real-time longitudinal safety."""

from __future__ import annotations

import cv2
import numpy as np

import control_params as cp
from live_bev_viewer import CLASS_STYLE, DEFAULT_STYLE


ZONE_COLORS = {"SAFE": (50, 190, 70), "CRITICAL": (20, 170, 240),
               "COLLISION": (30, 30, 230), "UNAVAILABLE": (100, 100, 100)}
TOP, LEFT, SCALE = 65, 55, 5
ROWS, COLS = 120, 80


def cell_center(row: int, col: int) -> tuple[int, int]:
    return LEFT + col * SCALE + 2, TOP + row * SCALE + 2


def render_realtime_frame(frame, timing: dict | None = None) -> np.ndarray:
    """Render actual full raw/cleaned matrices; boundaries come from result."""
    image = np.full((760, 1240, 3), 242, dtype=np.uint8)
    right = 520
    cv2.rectangle(image, (LEFT, TOP), (LEFT + COLS*SCALE, TOP + ROWS*SCALE),
                  (248, 248, 248), -1)
    result, safety = frame.live_result, frame.live_result.safety_result
    corridor_left = int(LEFT + (40-cp.WIDTH_OF_INTEREST_M/2)*SCALE)
    corridor_right = int(LEFT + (40+cp.WIDTH_OF_INTEREST_M/2)*SCALE)
    ego_y = TOP + 80*SCALE
    if safety is not None:
        def boundary_y(value):
            return int(round(TOP + (80-min(max(value, 0), 80))*SCALE))
        yc, yk = boundary_y(safety.collision_boundary_m), boundary_y(safety.critical_boundary_m)
        cv2.rectangle(image, (corridor_left, yc), (corridor_right, ego_y), (220,205,255), -1)
        cv2.rectangle(image, (corridor_left, yk), (corridor_right, yc), (205,235,250), -1)
        cv2.rectangle(image, (corridor_left, TOP), (corridor_right, yk), (215,245,220), -1)
    for row in range(0, ROWS+1, 10):
        y = TOP + row*SCALE
        cv2.line(image, (LEFT,y), (LEFT+COLS*SCALE,y), (220,220,220), 1)
    for col in range(0, COLS+1, 10):
        x = LEFT + col*SCALE
        cv2.line(image, (x,TOP), (x,TOP+ROWS*SCALE), (225,225,225), 1)
    cv2.rectangle(image, (LEFT,TOP), (LEFT+COLS*SCALE,TOP+ROWS*SCALE), (60,60,60), 2)
    cv2.rectangle(image, (corridor_left,TOP), (corridor_right,ego_y), (70,70,70), 1)
    cv2.line(image, (LEFT,ego_y), (LEFT+COLS*SCALE,ego_y), (100,100,100), 1)
    ego_x, _ = cell_center(80,40)
    cv2.line(image, (ego_x,TOP), (ego_x,TOP+ROWS*SCALE), (100,100,100), 1)
    cv2.circle(image, (ego_x,ego_y+2), 7, (20,20,20), -1)
    cv2.arrowedLine(image, (ego_x,ego_y+2), (ego_x,ego_y-20), (20,20,20), 2)
    cv2.putText(image, "LIVE FULL 120 x 80 BEV", (LEFT,35), cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),2)
    for row, col in zip(*np.where(frame.raw_matrix != 0)):
        cls = int(frame.raw_matrix[row,col]); name,color=CLASS_STYLE.get(cls,DEFAULT_STYLE)
        point=cell_center(int(row),int(col)); cv2.circle(image,point,6,color,-1)
        cv2.circle(image,point,6,(30,30,30),1)
        cv2.putText(image,name,(point[0]+7,point[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.34,(30,30,30),1)
    for row, col in zip(*np.where(result.cleaned_matrix != 0)):
        cv2.circle(image,cell_center(int(row),int(col)),9,(255,255,255),1)
    if safety is not None:
        for value,color,label in ((safety.collision_boundary_m,ZONE_COLORS["COLLISION"],"COLLISION"),
                                  (safety.critical_boundary_m,ZONE_COLORS["CRITICAL"],"CRITICAL")):
            if value <= 80:
                y=int(round(TOP+(80-value)*SCALE)); cv2.line(image,(corridor_left,y),(corridor_right,y),color,3)
            else:
                cv2.putText(image,f"{label} BOUNDARY > BEV HORIZON ({value:.1f} m)",
                            (LEFT+5,90 if label=="COLLISION" else 110),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2)
    zone=result.safety_zone; color=ZONE_COLORS.get(zone,ZONE_COLORS["UNAVAILABLE"])
    cv2.rectangle(image,(right,35),(1215,150),color,-1)
    cv2.putText(image,f"CURRENT ZONE: {zone}",(right+20,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),3)
    cv2.putText(image,f"SAFE TARGET SPEED: {result.safe_target_speed_mph:.2f} mph",
                (right+20,125),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    def fmt(value, suffix=""): return "unavailable" if value is None else f"{value:.3f}{suffix}"
    timing=timing or {}
    rows=(
        ("Frame / timestamp",f"{frame.frame_id} / {frame.sensor_timestamp_ns}"),
        ("GPS speed",fmt(frame.gps.speed_mps," m/s")),
        ("IMU yaw source",f"{frame.yaw_rate_source}: {fmt(frame.imu_yaw.tracker_vehicle_yaw_rate_rps,' rad/s')}"),
        ("IMU longitudinal accel",fmt(frame.imu_acceleration.vehicle_longitudinal_acceleration_mps2," m/s2")),
        ("Nearest safety gap",fmt(None if safety is None else safety.input_obstacle_gap_m," m")),
        ("Collision boundary",fmt(None if safety is None else safety.collision_boundary_m," m")),
        ("Critical boundary",fmt(None if safety is None else safety.critical_boundary_m," m")),
        ("Coverage","unavailable" if safety is None else safety.coverage_status),
        ("Nominal target",f"{result.nominal_target_speed_mph:.2f} mph"),
        ("Safe target",f"{result.safe_target_speed_mph:.2f} mph"),
        ("Raw / tracker dt",f"{fmt(frame.raw_dt_s,' s')} / {frame.tracker_dt_s:.3f} s"),
        ("dt clamped",str(frame.dt_clamped)),
        ("Tracker latency",f"{result.tracker_latency_ms:.2f} ms"),
        ("Full processing",f"{frame.processing_time_ms:.2f} ms"),
        ("Rolling p99",fmt(timing.get("p99")," ms")),
    )
    y=185
    for label,text in rows:
        cv2.putText(image,label,(right,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,80,80),1)
        cv2.putText(image,text,(right+225,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1); y+=32
    cv2.putText(image,"READ / TRACK / EVALUATE / DISPLAY / LOG ONLY",(right,700),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,160),2)
    cv2.putText(image,"Ctrl+C, q, or ESC to exit",(right,730),cv2.FONT_HERSHEY_SIMPLEX,.5,(70,70,70),1)
    return image
