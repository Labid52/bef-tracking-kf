#!/usr/bin/env python3
"""Archive equivalence and full recorded-frame timing for standalone runner."""
import csv,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
from real_longitudinal_safety_runner import RealLongitudinalSafetyRunner
from realtime_bev_renderer import render_realtime_frame

def percentile(a,q):
 a=sorted(a);x=(len(a)-1)*q;i=int(x);j=min(i+1,len(a)-1);return a[i]+(x-i)*(a[j]-a[i])
def summary(a):
 return {"mean":sum(a)/len(a),"median":percentile(a,.5),"p95":percentile(a,.95),"p99":percentile(a,.99),"max":max(a),"over_50_ms":sum(x>50 for x in a),"over_50_percent":100*sum(x>50 for x in a)/len(a)}
def main():
 archive=Path("matrix_imu_gps.zip");reference={int(r["frame_id"]):r for r in csv.DictReader(open("gps_imu_yaw_longitudinal_safety_replay.csv"))}
 runner=RealLongitudinalSafetyRunner();mismatch={"gps":0,"collision":0,"critical":0,"zone":0,"safe_target":0};internal=[];nonrender=[];render=[];full=[];longest=run=0
 with zipfile.ZipFile(archive) as z:
  records=[json.loads(x) for x in z.read("sensors/matrix_sensors.jsonl").splitlines()]
  for rec in records:
   cycle_start=time.perf_counter()
   matrix=np.load(io.BytesIO(z.read(rec["raw_matrix_file"])),allow_pickle=False)
   out=runner.process_sample(matrix,rec,rec["raw_matrix_file"]);internal.append(out.processing_time_ms);out.log_row();nonrender.append((time.perf_counter()-cycle_start)*1e3);ref=reference[out.frame_id];s=out.live_result.safety_result
   mismatch["gps"]+=not math.isclose(out.gps.speed_mps,float(ref["gps_speed_mps"]),abs_tol=1e-12)
   mismatch["collision"]+=not math.isclose(s.collision_boundary_m,float(ref["collision_boundary_m"]),abs_tol=1e-12)
   mismatch["critical"]+=not math.isclose(s.critical_boundary_m,float(ref["critical_boundary_m"]),abs_tol=1e-12)
   mismatch["zone"]+=s.zone!=ref["zone"]
   mismatch["safe_target"]+=not math.isclose(s.safe_target_speed_mph,float(ref["safe_target_speed_mph"]),abs_tol=1e-12)
   t=time.perf_counter();render_realtime_frame(out);render.append((time.perf_counter()-t)*1e3);full.append((time.perf_counter()-cycle_start)*1e3)
 for x in nonrender:
  run=run+1 if x>50 else 0;longest=max(longest,run)
 print(json.dumps({"frames":runner.processed_count,"mismatches":mismatch,"pipeline_internal_ms":summary(internal),"acquire_parse_process_log_ms":summary(nonrender),"render_only_ms":summary(render),"full_rendered_path_ms":summary(full),"non_render_longest_over_50_run":longest},indent=2))
if __name__=="__main__":main()
