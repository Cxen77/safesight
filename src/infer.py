"""Phase 1: pretrained PPE detection on a webcam, video, or image folder."""
import argparse
from collections import deque
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"person": (255, 255, 255), "helmet": (0, 255, 0),
          "head": (0, 0, 255), "vest": (0, 255, 255),
          "safety vest": (0, 255, 255)}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/sh17_yolov8n.pt")
    parser.add_argument("--source", default="0", help="Camera index, video, image, or image folder")
    parser.add_argument("--no-display", action="store_true", help="Process without a window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--output", type=Path, default=ROOT / "results/demo_output.mp4")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    camera = args.source.isdecimal()
    source = Path(args.source)
    if not camera and not source.exists():
        parser.error(f"Source not found: {source}")
    if not camera and source.resolve() == args.output.resolve():
        parser.error("Output must differ from the input video; choose --output")
    model = YOLO(str(args.weights))
    print(f"Model classes: {model.names}", flush=True)
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES) if source.is_dir() else []
    image_mode = source.is_dir() or source.suffix.lower() in IMAGE_TYPES
    if image_mode and not source.is_dir():
        images = [source]
    if image_mode and not images:
        parser.error(f"No supported images found in {source}")
    cap = writer = None
    durations, count = deque(maxlen=30), 0
    try:
        if not image_mode:
            cap = cv2.VideoCapture(int(args.source) if camera else str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source {args.source}; check connection and camera permissions")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_fps = video_fps if 0 < video_fps <= 240 else 30.0
            args.output.parent.mkdir(parents=True, exist_ok=True)
        else:
            image_out = ROOT / "results/demo_images"
            image_out.mkdir(parents=True, exist_ok=True)
        while True:
            start = time.perf_counter()
            if image_mode:
                if count >= len(images):
                    break
                frame = cv2.imread(str(images[count]))
                if frame is None:
                    raise RuntimeError(f"Cannot read image: {images[count]}")
            else:
                ok, frame = cap.read()
                if not ok:
                    if count == 0 or camera:
                        raise RuntimeError("No frame received from source")
                    break
            result = model.predict(frame, imgsz=640, verbose=False)[0]
            people = 0
            for box in result.boxes:
                label = model.names[int(box.cls.item())]
                key = label.lower().replace("-", " ").replace("_", " ")
                people += key == "person"
                color = COLORS.get(key, (160, 160, 160))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {box.conf.item():.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            durations.append(time.perf_counter() - start)
            fps = len(durations) / max(sum(durations), 1e-9)
            cv2.rectangle(frame, (0, 0), (330, 38), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f} | People: {people}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if image_mode:
                if not cv2.imwrite(str(image_out / f"{count:05d}_{images[count].stem}.jpg"), frame):
                    raise RuntimeError("Could not save annotated image")
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video: {args.output}")
                writer.write(frame)
            count += 1
            if not args.no_display:
                cv2.imshow("PPE demo - q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and count >= args.max_frames:
                break
        print(f"Processed {count} frames. Saved to {image_out if image_mode else args.output}")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
