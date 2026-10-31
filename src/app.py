"""Local safety dashboard. Use --demo to exercise alerts without camera/models."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, send_from_directory

from alerts import AlertDriver, AlertPolicy, EventStore
from dashboard import DASHBOARD_HTML
from risk import compute_risk
from signals import FireHistory, associate_ppe, normalize_inputs
from zones import DEFAULT_ZONE, ZoneMonitor

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    demo: bool = False
    source: str = "0"
    ppe_weights: Path = ROOT / "models/ppe_best.pt"
    fire_weights: Path = ROOT / "models/fire_best.pt"
    zones: Path = DEFAULT_ZONE
    device: str = "cpu"
    hardware: bool = False
    silent: bool = False
    output_dir: Path | None = None
    max_frames: int | None = None


class SafetyPipeline:
    def __init__(self, settings):
        self.settings = settings
        output = settings.output_dir or (ROOT / "results/demo" if settings.demo else ROOT / "results")
        self.store, self.policy = EventStore(output), AlertPolicy()
        self.condition, self.stop_event = threading.Condition(), threading.Event()
        self.thread, self.jpeg, self.sequence = None, None, 0
        self.state = {"mode": "demo" if settings.demo else "live", "status": "starting", "error": None,
            "score": None, "band": None, "raw_score": None, "raw_band": None,
            "fps": None, "latency_ms": None, "people": None, "crowd_count": None,
            "ppe_load": None, "zone_pressure": None, "proximity": None, "fire_evidence": None,
            "latched": False, "pending": False, "output_level": "Low", "cause": "Waiting for the first frame",
            "updated_at": None, "has_frame": False, "detectors_active": False}

    def publish(self, frame=None, **values):
        jpeg = None
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise RuntimeError("Unable to encode camera frame")
            jpeg = encoded.tobytes()
        with self.condition:
            self.state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            if jpeg is not None:
                self.jpeg, self.state["has_frame"] = jpeg, True
                self.sequence += 1
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            snapshot = deepcopy(self.state)
        snapshot["events"] = self.store.recent()
        return snapshot

    def frames(self):
        last_sequence = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: (self.jpeg is not None and self.sequence != last_sequence)
                                        or self.stop_event.is_set(), timeout=1)
                if self.sequence == last_sequence or self.jpeg is None:
                    continue
                last_sequence, jpeg = self.sequence, self.jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Pipeline has already been started")
        self.thread = threading.Thread(target=self.run, name="safesight-pipeline", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)

    def demo_frame(self, elapsed):
        t = elapsed % 32
        inside, helmet, vest, count, fire, bare_seconds = False, True, True, 1, 0.0, 0.0
        longest = 0.0
        if 4 <= t < 10:
            inside, helmet, vest, longest = True, False, False, t-4
            bare_seconds = longest
        elif 10 <= t < 14:
            inside, helmet, vest, fire, longest = True, False, False, .85, t-4
            bare_seconds = longest
        elif 14 <= t < 20 or t >= 26:
            count = 0
        elif 20 <= t < 26:
            inside, vest, count, longest = True, False, 2, t-20+5
        crowd = count if inside else 0
        values = normalize_inputs(count, count if not helmet else 0, count if not vest else 0,
                                  crowd, longest, 0 if inside else 1, fire)
        frame = np.full((540, 960, 3), (218, 225, 228), dtype=np.uint8)
        cv2.rectangle(frame, (0, 290), (960, 540), (175, 190, 198), -1)
        for x in (80, 340, 600, 860):
            cv2.line(frame, (x, 45), (x, 290), (183, 199, 207), 2)
        polygon = np.array([[450,210],[880,210],[930,480],[400,480]], dtype=np.int32)
        cv2.polylines(frame, [polygon], True, (25, 128, 220), 3)
        cv2.putText(frame, "Restricted zone", (580, 470), cv2.FONT_HERSHEY_SIMPLEX, .6, (35,70,100), 2)
        for index in range(count):
            x = 570 + index*155 if inside else 150
            cv2.rectangle(frame, (x,135), (x+90,440), (110,92,65), 2)
            cv2.circle(frame, (x+45,175), 22, (130,170,200), -1)
            cv2.line(frame, (x+45,202), (x+45,340), (100,75,50), 28)
            cv2.line(frame, (x+45,330), (x+20,432), (100,75,50), 14)
            cv2.line(frame, (x+45,330), (x+70,432), (100,75,50), 14)
            if helmet:
                cv2.ellipse(frame, (x+45,161), (27,16), 0, 180, 360, (25,185,235), -1)
            if vest:
                cv2.rectangle(frame, (x+24,210), (x+66,295), (50,185,210), -1)
            cv2.putText(frame, f"Person {index+1}", (x-3,125), cv2.FONT_HERSHEY_SIMPLEX, .5, (45,65,80), 1)
        if fire:
            cv2.fillPoly(frame, [np.array([[810,300],[860,205],[900,300]])], (40,105,225))
            cv2.putText(frame, "Simulated fire", (735,325), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,60,130), 2)
        cv2.putText(frame, "DEMO SIMULATION - no camera or AI detections", (25,34), cv2.FONT_HERSHEY_SIMPLEX, .65, (45,65,80), 2)
        cause = "Scripted scene clear" if not count else ("Simulated fire evidence" if fire else
                 f"Simulated bare head in zone {bare_seconds:.1f}s" if not helmet else
                 "Simulated crowd in restricted zone" if crowd else "Simulated worker outside zone")
        return frame, values, bare_seconds, count, crowd, cause

    def run(self):
        cap = driver = None
        try:
            driver = AlertDriver(self.settings.hardware, audible=not self.settings.silent)
            checks, bare_entered, monitor = FireHistory(), {}, None
            times, frame_number, started = deque(maxlen=30), 0, time.monotonic()
            if not self.settings.demo:
                from ultralytics import YOLO
                ppe, fire_model = YOLO(str(self.settings.ppe_weights)), YOLO(str(self.settings.fire_weights))
                ppe_names = {str(name).lower().replace("-", " ").replace("safety vest", "vest") for name in ppe.names.values()}
                if not {"person", "head", "helmet", "vest"}.issubset(ppe_names):
                    raise ValueError("PPE model must declare person, head, helmet, and vest classes")
                if not {"fire", "smoke"}.issubset({str(name).lower() for name in fire_model.names.values()}):
                    raise ValueError("Fire model must declare fire and smoke classes")
                cap = cv2.VideoCapture(int(self.settings.source) if self.settings.source.isdecimal() else self.settings.source)
                if not cap.isOpened():
                    raise RuntimeError("Cannot open camera/video source")
            while not self.stop_event.is_set():
                tick = time.perf_counter()
                if self.settings.demo:
                    frame, values, bare_seconds, people, crowd, cause = self.demo_frame(time.monotonic()-started)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera stopped providing frames or video ended")
                    if monitor is None:
                        monitor = ZoneMonitor((frame.shape[1], frame.shape[0]), self.settings.zones)
                    detection = ppe.track(frame, imgsz=640, persist=True, device=self.settings.device,
                                          tracker="bytetrack.yaml", verbose=False)[0]
                    persons, equipment = [], []
                    for box in detection.boxes:
                        name = str(ppe.names[int(box.cls.item())]).lower()
                        coordinates = tuple(map(float, box.xyxy[0].tolist()))
                        if name == "person":
                            persons.append({"id": int(box.id.item()) if box.id is not None else None, "box": coordinates})
                        else:
                            equipment.append({"name": name, "box": coordinates})
                    if frame_number % 10 == 0:
                        result = fire_model.predict(frame, imgsz=640, device=self.settings.device, verbose=False)[0]
                        evidence = [float(box.conf.item()) for box in result.boxes
                                    if str(fire_model.names[int(box.cls.item())]).lower() in ("fire", "smoke")]
                        checks.add(max(evidence, default=0.0))
                    people = len(persons)
                    if any(person["id"] is None for person in persons):
                        bare_entered.clear()
                        monitor.update([], frame.shape)
                        self.policy.interrupt_observation()
                        self.publish(frame=frame, status="warming", error="Waiting for stable person tracking")
                        frame_number += 1
                        if self.settings.max_frames and frame_number >= self.settings.max_frames:
                            break
                        continue
                    associated = associate_ppe(persons, equipment)
                    zone = monitor.update([(p["id"], *p["box"]) for p in persons], frame.shape)
                    crowd = zone["crowd_count"]
                    now = time.monotonic()
                    bare_ids = {p["id"] for p in associated if p["bare_head"] and p["id"] in zone["intruders"]}
                    bare_entered = {track_id: bare_entered.get(track_id, now) for track_id in bare_ids}
                    bare_seconds = max((now-entered for entered in bare_entered.values()), default=0.0)
                    missing_helmet = sum(not p["helmet"] for p in associated)
                    missing_vest = sum(not p["vest"] for p in associated)
                    values = normalize_inputs(people, missing_helmet, missing_vest, crowd,
                        max(zone["duration_s"].values(), default=0.0), zone["proximity_margin"], checks.evidence)
                    cause = (f"Fire/smoke evidence {checks.evidence:.2f}" if checks.evidence >= .7 else
                             f"Observed bare head in zone {bare_seconds:.1f}s" if bare_seconds else
                             f"{crowd} people inside restricted zone" if crowd else
                             f"Helmet not detected for {missing_helmet}; vest not detected for {missing_vest}" if missing_helmet or missing_vest else "Scene clear")
                    frame = detection.plot()
                    cv2.polylines(frame, [monitor.polygon(frame.shape).astype(np.int32)], True, (25,128,220), 2)
                score, band = compute_risk(**values, unhelmeted_in_zone_seconds=bare_seconds)
                latency = (time.perf_counter()-tick)*1000
                decision = self.policy.update(score, band)
                driver.set_level(decision.output_level)
                color = {"Low":(45,105,30),"Medium":(20,105,130),"High":(20,70,145),"Critical":(45,45,150)}[decision.band]
                cv2.rectangle(frame, (0,frame.shape[0]-42), (frame.shape[1],frame.shape[0]), color, -1)
                cv2.putText(frame, f"{decision.band}  {decision.score:.1f}/100" + ("  LATCHED" if decision.latched else ""),
                            (18,frame.shape[0]-13), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
                if decision.event_band:
                    self.store.record(decision.event_band, score, values, cause, frame)
                times.append(time.perf_counter())
                fps = (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
                self.publish(frame=frame, status="active", error=None, score=decision.score, band=decision.band,
                    raw_score=score, raw_band=band, fps=fps, latency_ms=latency, people=people, crowd_count=crowd,
                    cause=cause, latched=decision.latched, pending=decision.pending,
                    output_level=decision.output_level, detectors_active=not self.settings.demo, **values)
                frame_number += 1
                if self.settings.max_frames and frame_number >= self.settings.max_frames:
                    break
                if self.settings.demo:
                    self.stop_event.wait(.2)
        except Exception as error:
            self.publish(status="error", error=str(error), score=None, band=None, detectors_active=False)
        finally:
            if cap is not None:
                cap.release()
            if driver is not None:
                try:
                    driver.close()
                except Exception as error:
                    self.publish(status="error", error=f"Alert output cleanup failed: {error}", score=None, band=None)
            if self.snapshot()["status"] != "error":
                self.publish(status="stopped", score=None, band=None, detectors_active=False)
            self.stop_event.set()
            with self.condition:
                self.condition.notify_all()


def create_app(pipeline):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.get("/api/state")
    def state():
        response = jsonify(pipeline.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/video_feed")
    def video_feed():
        return Response(pipeline.frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/snapshots/<path:filename>")
    def snapshot(filename):
        return send_from_directory(pipeline.store.snapshots, filename)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--source", default="0")
    parser.add_argument("--ppe-weights", type=Path, default=ROOT / "models/ppe_best.pt")
    parser.add_argument("--fire-weights", type=Path, default=ROOT / "models/fire_best.pt")
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONE)
    parser.add_argument("--device", default="cpu", help="Use cpu, or 0 once GPU training is stopped")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if args.hardware and args.demo:
        parser.error("Hardware outputs require live mode")
    if not args.demo:
        for path in (args.ppe_weights, args.fire_weights, args.zones):
            if not path.is_file():
                parser.error(f"Missing live-pipeline file: {path}. Use --demo while models/zone are unavailable.")
    pipeline = SafetyPipeline(Settings(**{key:value for key,value in vars(args).items() if key != "port"}))
    app = create_app(pipeline)
    pipeline.start()
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
