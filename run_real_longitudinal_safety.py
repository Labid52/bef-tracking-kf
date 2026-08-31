#!/usr/bin/env python3
"""Standalone read-only consumer of live BEV matrices and sensor JSONL."""

from __future__ import annotations
import argparse,csv,time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
from real_longitudinal_safety_runner import LiveFileSampleSource,RealLongitudinalSafetyRunner
from realtime_bev_renderer import render_realtime_frame

FIELDS=("frame","timestamp_ns","gps_speed_mps","gps_valid","speed_source",
"imu_yaw_rate_rps","yaw_rate_source","imu_longitudinal_acceleration_mps2",
"imu_acceleration_valid","bev_obstacle_gap_m","collision_boundary_m",
"critical_boundary_m","zone","coverage_status","nominal_target_speed_mph",
"safe_target_speed_mph","raw_dt_s","tracker_dt_s","dt_clamped",
"tracker_latency_ms","processing_time_ms")

def stats(values):
    if not values:return {"mean":0,"median":0,"p95":0,"p99":0,"max":0}
    a=sorted(values)
    def p(q):
        x=(len(a)-1)*q;i=int(x);j=min(i+1,len(a)-1);return a[i]+(x-i)*(a[j]-a[i])
    return {"mean":sum(a)/len(a),"median":p(.5),"p95":p(.95),"p99":p(.99),"max":a[-1]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--matrix-dir",type=Path,default=Path("matrix"))
    ap.add_argument("--sensor-file",type=Path,default=Path("sensors/matrix_sensors.jsonl"))
    ap.add_argument("--start",default="latest",help="latest, earliest, or frame ID")
    ap.add_argument("--poll-ms",type=float,default=10.0)
    ap.add_argument("--headless",action="store_true")
    ap.add_argument("--log",type=Path)
    ap.add_argument("--no-log",action="store_true")
    args=ap.parse_args(); start=int(args.start) if args.start.isdigit() else args.start
    source=LiveFileSampleSource(args.matrix_dir,args.sensor_file,start=start)
    runner=RealLongitudinalSafetyRunner(); samples=deque(maxlen=2048)
    log_path=None if args.no_log else (args.log or Path("logs")/f"realtime_safety_{datetime.now():%Y%m%d_%H%M%S}.csv")
    handle=writer=None
    if log_path:
        log_path.parent.mkdir(parents=True,exist_ok=True);handle=log_path.open("w",newline="");writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
    stopped="Ctrl+C"
    try:
        while True:
            ready=source.poll_ready(limit=32)
            if not ready:
                time.sleep(max(args.poll_ms,1)/1000);continue
            for frame_id,path,matrix,record in ready:
                frame=runner.process_sample(matrix,record,path);samples.append(frame.processing_time_ms)
                if writer:writer.writerow(frame.log_row());handle.flush()
                if not args.headless:
                    cv2.imshow("Standalone Real-Time Longitudinal Safety",render_realtime_frame(frame,stats(samples)))
                    if cv2.waitKey(1)&0xff in (27,ord('q')):stopped="window key";return
    except KeyboardInterrupt: pass
    finally:
        if handle:handle.flush();handle.close()
        cv2.destroyAllWindows();s=stats(samples)
        print(f"Stopped by {stopped}; processed={runner.processed_count}; log={log_path}")
        print("Processing ms " + " ".join(f"{k}={v:.2f}" for k,v in s.items()))

if __name__=="__main__":main()
