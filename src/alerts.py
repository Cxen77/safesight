"""Debounced alert policy, change-only outputs, and snapshot-backed event logs."""
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import math
from pathlib import Path
import threading
import time
import uuid

import cv2
from risk import risk_band

BANDS = ("Low", "Medium", "High", "Critical")
INPUT_NAMES = ("ppe_load", "zone_pressure", "proximity", "fire_evidence")
EVENT_COLUMNS = ("timestamp", "band", "score", *INPUT_NAMES, "cause")


@dataclass(frozen=True)
class AlertDecision:
    score: float
    band: str
    output_level: str
    event_band: str | None
    latched: bool
    pending: bool


class AlertPolicy:
    """Confirm >=High for 1s; latch confirmed Critical until 3s below High.

    A High-to-Critical change does not reset the shared >=High debounce timer.
    Log once per alarm episode, with one additional event for Critical escalation.
    """
    def __init__(self, clock=None):
        self.clock = clock or time.monotonic
        self.high_since = self.clear_since = None
        self.latched = False
        self.critical_score = 0.0
        self.logged_rank = 0

    def interrupt_observation(self):
        """Unknown frames cannot confirm an alert or count toward latch clearing."""
        self.high_since = self.clear_since = None
        if not self.latched:
            self.logged_rank = 0

    def update(self, score, band):
        if band not in BANDS or risk_band(score) != band:
            raise ValueError("Band must match the finite 0–100 risk score")
        now, event = self.clock(), None
        alarming = band in ("High", "Critical")
        if alarming:
            self.clear_since = None
            if self.high_since is None:
                self.high_since = now
            confirmed = now - self.high_since >= 1.0
            if confirmed:
                if band == "Critical":
                    self.latched = True
                    self.critical_score = max(self.critical_score, score)
                rank = BANDS.index(band)
                if rank > self.logged_rank:
                    event, self.logged_rank = band, rank
        else:
            self.high_since = None
            confirmed = False
            if self.latched:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= 3.0:
                    self.latched = False
                    self.critical_score = 0.0
                    self.logged_rank = 0
            else:
                self.logged_rank = 0
        if self.latched:
            return AlertDecision(max(score, self.critical_score), "Critical", "Critical", event, True, False)
        output = band if not alarming or confirmed else "Low"
        return AlertDecision(score, band, output, event, False, alarming and not confirmed)


class AlertDriver:
    def __init__(self, hardware=False, audible=True):
        self.hardware, self.audible = hardware, audible
        self.level, self.closed, self.gpio = None, False, None
        if hardware:
            try:
                self.gpio = importlib.import_module("Jetson.GPIO")
                self.gpio.setmode(self.gpio.BOARD)
                self.gpio.setup((12, 16), self.gpio.OUT, initial=self.gpio.LOW)
            except Exception as error:
                if self.gpio is not None:
                    self.gpio.cleanup((12, 16))
                raise RuntimeError("Cannot initialize Jetson GPIO pins 12/16") from error

    def set_level(self, band):
        if self.closed or band not in BANDS:
            raise ValueError("Driver is closed or alert band is invalid")
        if band == self.level:
            return
        alarming = band in ("High", "Critical")
        if self.hardware:
            self.gpio.output((12, 16), self.gpio.HIGH if alarming else self.gpio.LOW)
        elif alarming and self.audible:
            try:
                sound = importlib.import_module("winsound")
                sound.PlaySound("SystemExclamation", sound.SND_ALIAS | sound.SND_ASYNC)
            except (ImportError, RuntimeError):
                print("BUZZER ON", flush=True)
        self.level = band
        print(f"Alert level: {band}", flush=True)

    def close(self):
        if self.closed:
            return
        if self.gpio is not None:
            try:
                self.gpio.output((12, 16), self.gpio.LOW)
            finally:
                self.gpio.cleanup((12, 16))
        self.closed = True


class EventStore:
    def __init__(self, root):
        self.root, self.lock = Path(root), threading.Lock()
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "events.csv"
        self.events = []
        if self.csv_path.exists() and self.csv_path.stat().st_size:
            with self.csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
                    raise ValueError("Event CSV header does not match this project")
                for row in deque(reader, maxlen=10):
                    stamp = datetime.fromisoformat(row["timestamp"])
                    matches = sorted(self.snapshots.glob(stamp.strftime("%Y%m%dT%H%M%S%fZ_") + "*.jpg"))
                    self.events.append({**row, "score": float(row["score"]),
                        **{name: float(row[name]) for name in INPUT_NAMES},
                        "snapshot": matches[0].name if matches else None})

    def record(self, band, score, inputs, cause, frame):
        if band not in ("High", "Critical") or risk_band(score) != band:
            raise ValueError("Only valid High/Critical events can be recorded")
        values = {name: float(inputs[name]) for name in INPUT_NAMES}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("Event inputs must be finite normalized numbers")
        now = datetime.now(timezone.utc)
        key = now.strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex[:8]
        snapshot = self.snapshots / f"{key}.jpg"
        row = {"timestamp": now.isoformat(), "band": band, "score": float(score), **values, "cause": cause}
        with self.lock:
            if not cv2.imwrite(str(snapshot), frame):
                raise RuntimeError("Could not write event snapshot")
            try:
                needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=EVENT_COLUMNS)
                    if needs_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception:
                snapshot.unlink(missing_ok=True)
                raise
            self.events.append({**row, "snapshot": snapshot.name})
            self.events = self.events[-10:]
            return self.events[-1].copy()

    def recent(self):
        with self.lock:
            return [event.copy() for event in reversed(self.events)]
