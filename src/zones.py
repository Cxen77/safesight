"""Resolution-independent zone editing and tracked-person intrusion monitoring."""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONE = Path(__file__).with_name("zones.json")


def validate_points(points):
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("A zone needs at least three x/y points")
    if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
        raise ValueError("Zone coordinates must be finite fractions between 0 and 1")
    if len(set(map(tuple, polygon))) != len(polygon):
        raise ValueError("Zone points must not repeat")
    if cv2.contourArea(polygon.astype(np.float32)) < 1e-8:
        raise ValueError("Zone must enclose an area")
    def cross(a, b, c):
        ab, ac = b - a, c - a
        return ab[0] * ac[1] - ab[1] * ac[0]
    def on_segment(a, b, c):
        return abs(cross(a, b, c)) < 1e-10 and np.all(c >= np.minimum(a, b) - 1e-10) and np.all(c <= np.maximum(a, b) + 1e-10)
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        for j in range(i + 1, len(polygon)):
            if j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            c, d = polygon[j], polygon[(j + 1) % len(polygon)]
            if (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0) or any(
                    (on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b))):
                raise ValueError("Zone edges must not cross or touch themselves")
    return polygon.astype(np.float32)


def draw_zone(image_path_or_frame, save_path=DEFAULT_ZONE, preview_path=ROOT / "results/zone_preview.png"):
    frame = (cv2.imread(str(image_path_or_frame)) if isinstance(image_path_or_frame, (str, Path))
             else np.asarray(image_path_or_frame).copy())
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Provide a readable color image or BGR frame")
    height, width = frame.shape[:2]
    points, window = [], "Zone editor: left add | right undo | Enter save | Esc cancel"
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < width and 0 <= y < height:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            preview = frame.copy()
            if points:
                cv2.polylines(preview, [np.array(points, dtype=np.int32)], len(points) >= 3, (0, 200, 255), 2)
                for point in points:
                    cv2.circle(preview, point, 4, (0, 200, 255), -1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (10, 13):
                fractions = [[x / width, y / height] for x, y in points]
                try:
                    validate_points(fractions)
                except ValueError as error:
                    print(f"Cannot save zone: {error}", flush=True)
                    continue
                save_path, preview_path = Path(save_path), Path(preview_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError(f"Could not save zone preview: {preview_path}")
                save_path.write_text(json.dumps({"points": fractions}, indent=2), encoding="utf-8")
                print(f"Saved zone to {save_path}", flush=True)
                return fractions
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


class ZoneMonitor:
    """Safe margin: 0 inside/on boundary, 1 at least 15% frame width outside.

    Duration is continuous observed presence; leaving or disappearing resets it.
    The frame size is (width, height); update accepts OpenCV's frame.shape.
    """
    def __init__(self, frame_size, zones_path=DEFAULT_ZONE, clock=None):
        self.width, self.height = map(int, frame_size)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive")
        config = json.loads(Path(zones_path).read_text(encoding="utf-8"))
        self.points = validate_points(config["points"])
        self.clock, self.entered = clock or time.monotonic, {}

    def polygon(self, frame_shape=None):
        height, width = frame_shape[:2] if frame_shape is not None else (self.height, self.width)
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        return (self.points * np.array([width, height], dtype=np.float32)).reshape(-1, 1, 2)

    def update(self, person_boxes_with_ids, frame_shape):
        polygon = self.polygon(frame_shape)
        now, inside, seen, distances = self.clock(), [], set(), {}
        for track_id, x1, y1, x2, y2 in person_boxes_with_ids:
            if not all(math.isfinite(float(v)) for v in (track_id, x1, y1, x2, y2)):
                raise ValueError("Track IDs and box coordinates must be finite numbers")
            if int(track_id) != track_id or track_id in seen or x2 < x1 or y2 < y1:
                raise ValueError("Expected unique integer track IDs and ordered box coordinates")
            track_id = int(track_id)
            seen.add(track_id)
            distance = cv2.pointPolygonTest(polygon, ((float(x1) + float(x2)) / 2, float(y2)), True)
            distances[track_id] = float(distance)
            if distance >= 0:  # Include the boundary in the danger zone.
                inside.append(track_id)
        self.entered = {track_id: self.entered.get(track_id, now) for track_id in inside}
        durations = {track_id: max(0.0, now - self.entered[track_id]) for track_id in inside}
        closest = min((max(0.0, -distance) for distance in distances.values()), default=math.inf)
        margin = min(1.0, closest / (0.15 * frame_shape[1]))
        return {"intruders": inside, "duration_s": durations, "crowd_count": len(inside),
                "proximity_margin": margin, "signed_distance_px": distances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index or video file")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/ppe_best.pt")
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONE)
    parser.add_argument("--draw", action="store_true", help="Draw a polygon on the first captured frame")
    args = parser.parse_args()
    if not args.draw and (not args.weights.is_file() or not args.zones.is_file()):
        parser.error("Need model weights and a zone file; use --draw to configure a zone first")
    cap = cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot capture a frame from {args.source}")
        if args.draw:
            draw_zone(frame, args.zones)
            return
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        monitor = ZoneMonitor((frame.shape[1], frame.shape[0]), args.zones)
        next_print = 0.0
        while ok:
            result = model.track(frame, imgsz=640, persist=True, verbose=False)[0]
            people = []
            if result.boxes.id is not None:
                for box in result.boxes:
                    if str(model.names[int(box.cls.item())]).lower() == "person":
                        people.append((int(box.id.item()), *map(float, box.xyxy[0].tolist())))
            state = monitor.update(people, frame.shape)
            display = result.plot()
            cv2.polylines(display, [monitor.polygon(frame.shape).astype(np.int32)], True, (0, 200, 255), 2)
            if time.monotonic() >= next_print:
                print(json.dumps(state), flush=True)
                next_print = time.monotonic() + 1
            cv2.imshow("Zone monitor - q to quit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            ok, frame = cap.read()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
