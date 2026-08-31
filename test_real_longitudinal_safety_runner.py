#!/usr/bin/env python3
import io,json,time
from pathlib import Path
import numpy as np
import pytest

from real_longitudinal_safety_runner import (
    CompleteJSONLTailer,LiveFileSampleSource,RealLongitudinalSafetyRunner,
)
from realtime_bev_renderer import LEFT,SCALE,TOP,cell_center,render_realtime_frame

def record(frame,ns,speed=8.0,valid=True):
    return {"frame_id":frame,"raw_matrix_file":f"matrix/{frame:06d}.npy","monotonic_ns":ns,
      "gps":{"connected":valid,"received":valid,"fix":valid,"latitude_deg":0.0,"longitude_deg":0.0,
             "speed_kph":speed*3.6,"updated_monotonic_ns":ns,"age_ns":0,"satellites":12,"error":None},
      "imu":{"connected":True,"roll_deg":0.0,"pitch_deg":0.0,"yaw_deg":0.0,
             "acceleration_mps2":[1.25,0.0,9.80665],"angular_rate_rps":[0.0,0.0,-0.2],
             "updated_monotonic_ns":ns,"age_ns":0,"error":None}}

def matrix(gap=None):
    m=np.zeros((120,80),dtype=np.uint8)
    if gap:m[80-gap,40]=1
    return m

def save_bytes(m):
    b=io.BytesIO();np.save(b,m);return b.getvalue()

def test_exact_contract_persistent_processor_dt_speed_yaw_and_acceleration():
    runner=RealLongitudinalSafetyRunner();identity=id(runner.processor)
    a=runner.process_sample(matrix(),record(0,1_000_000_000,8.0))
    b=runner.process_sample(matrix(20),record(1,1_050_000_000,9.0))
    assert id(runner.processor)==identity and runner.processed_count==2
    assert a.raw_dt_s is None and b.raw_dt_s==.05 and b.tracker_dt_s==.05
    assert b.live_result.safety_input_ego_speed_mps==9.0
    assert b.imu_yaw.tracker_vehicle_yaw_rate_rps==.2
    assert b.live_result.ego_yaw_rate_rps==.2
    assert b.imu_acceleration.vehicle_longitudinal_acceleration_mps2==1.25

def test_shape_dtype_duplicate_and_timestamp_validation():
    r=RealLongitudinalSafetyRunner()
    with pytest.raises(ValueError):r.process_sample(np.zeros((1,1),dtype=np.uint8),record(0,1))
    with pytest.raises(TypeError):r.process_sample(np.zeros((120,80)),record(0,1))
    r.process_sample(matrix(),record(0,10))
    with pytest.raises(ValueError):r.process_sample(matrix(),record(0,20))
    with pytest.raises(ValueError):r.process_sample(matrix(),record(1,9))

def test_invalid_gps_is_unavailable_not_zero_and_imu_fallback():
    rec=record(0,10,valid=False);rec["imu"]["connected"]=False
    out=RealLongitudinalSafetyRunner().process_sample(matrix(),rec)
    assert out.speed_source=="GPS_UNAVAILABLE"
    assert out.live_result.safety_zone=="UNAVAILABLE"
    assert out.live_result.safety_input_ego_speed_mps is None
    assert out.yaw_rate_source=="TRACKER_ESTIMATE"

def test_safe_critical_collision_semantics_unchanged():
    for gap,zone in ((None,"SAFE"),(20,"CRITICAL"),(3,"COLLISION")):
        out=RealLongitudinalSafetyRunner().process_sample(matrix(gap),record(0,10,8.0))
        assert out.live_result.safety_zone==zone
        safe=out.live_result.safe_target_speed_mph;nom=out.live_result.nominal_target_speed_mph
        if zone=="SAFE":assert safe==nom
        elif zone=="COLLISION":assert safe==0
        else:assert 0<safe<=nom

def test_partial_json_line_is_not_parsed(tmp_path):
    p=tmp_path/"s.jsonl";payload=json.dumps(record(3,30)).encode()
    p.write_bytes(payload[:20]);tail=CompleteJSONLTailer(p);tail.poll();assert not tail.records
    with p.open("ab") as h:h.write(payload[20:]+b"\n")
    tail.poll();assert tail.pop(3)["frame_id"]==3

def test_matrix_before_sensor_partial_matrix_and_exact_pair(tmp_path):
    d=tmp_path/"matrix";d.mkdir();s=tmp_path/"s.jsonl";s.write_text("")
    source=LiveFileSampleSource(d,s,start="earliest")
    payload=save_bytes(matrix(10));p=d/"000000.npy";p.write_bytes(payload[:50])
    assert source.poll_ready()==[];assert source.poll_ready()==[]
    p.write_bytes(payload);assert source.poll_ready()==[]
    with s.open("a") as h:h.write(json.dumps(record(0,10))+"\n")
    ready=source.poll_ready();assert len(ready)==1 and ready[0][0]==0

def test_sensor_before_matrix_missing_frame_stall_and_no_duplicates(tmp_path):
    d=tmp_path/"matrix";d.mkdir();s=tmp_path/"s.jsonl"
    s.write_text(json.dumps(record(2,20))+"\n")
    source=LiveFileSampleSource(d,s,start="earliest")
    assert source.poll_ready()==[]
    np.save(d/"000002.npy",matrix());assert source.poll_ready()==[]
    ready=source.poll_ready();assert [x[0] for x in ready]==[2]
    assert source.poll_ready()==[]

def test_older_matrix_waits_for_exact_sensor_before_newer_pair(tmp_path):
    d=tmp_path/"matrix";d.mkdir();s=tmp_path/"s.jsonl"
    np.save(d/"000000.npy",matrix());np.save(d/"000001.npy",matrix())
    s.write_text(json.dumps(record(1,20))+"\n")
    source=LiveFileSampleSource(d,s,start="earliest",pair_wait_s=1.0)
    assert source.poll_ready()==[];assert source.poll_ready()==[]
    with s.open("a") as h:h.write(json.dumps(record(0,10))+"\n")
    ready=source.poll_ready();assert [x[0] for x in ready]==[0]
    ready=source.poll_ready();assert [x[0] for x in ready]==[1]

def test_three_hundred_incremental_frames_once_with_write_order_changes(tmp_path):
    d=tmp_path/"matrix";d.mkdir();s=tmp_path/"s.jsonl";s.write_text("")
    source=LiveFileSampleSource(d,s,start="earliest",max_pending=512);seen=[]
    for f in range(300):
        p=d/f"{f:06d}.npy";line=json.dumps(record(f,1_000_000_000+f*50_000_000))+"\n"
        if f%2: s.open("a").write(line);np.save(p,matrix(20 if f%3==0 else None))
        else: np.save(p,matrix(20 if f%3==0 else None));s.open("a").write(line)
        source.poll_ready();seen.extend(x[0] for x in source.poll_ready())
    assert seen==list(range(300)) and len(seen)==len(set(seen))

def test_full_renderer_outside_corridor_and_corridor_only_shading():
    m=matrix(20);m[100,5]=1
    out=RealLongitudinalSafetyRunner().process_sample(m,record(0,10,8.0))
    image=render_realtime_frame(out)
    assert image.shape==(760,1240,3) and out.raw_matrix.shape==(120,80)
    p=cell_center(100,5);assert tuple(image[p[1],p[0]]) != (248,248,248)
    y=TOP+51*SCALE+2
    assert tuple(image[y,LEFT+20*SCALE+2])==(248,248,248)
    assert tuple(image[y,LEFT+40*SCALE+2])!=(248,248,248)

def test_no_actuation_api_and_log_row_is_bounded():
    r=RealLongitudinalSafetyRunner();out=r.process_sample(matrix(),record(0,10))
    assert not any(hasattr(r,name) for name in ("throttle","brake","steering","can_write"))
    assert len(out.log_row())==21
