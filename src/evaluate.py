"""Measure detector maps, full-pipeline video speed, and expert scenario agreement."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BANDS = ("Low", "Medium", "High", "Critical")
SCENARIO_FIELDS = ("scenario_id", "ppe_load", "zone_pressure", "proximity",
                   "fire_evidence", "unhelmeted_in_zone_seconds", "expected_band")


def measured_at():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()


def write_final_numbers(machine):
    lines = ["# Final measured results", "", f"Updated: {measured_at()}",
             f"Machine: {machine}", ""]
    for label, path in (("PPE", ROOT / "results/ppe_eval.json"),
                        ("Fire/smoke", ROOT / "results/fire_eval.json")):
        if not path.is_file():
            lines += [f"## {label}", "", "Not measured yet.", ""]
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines += [f"## {label}", "", f"Evaluation machine: {data['machine']}",
                  f"Test mAP@0.50: {data['map50']:.4f}",
                  f"Test mAP@0.50:0.95: {data['map50_95']:.4f}", "",
                  "| Class | Precision | Recall | mAP@0.50 |", "| --- | ---: | ---: | ---: |"]
        for row in data["per_class"]:
            values = ["N/A" if row[key] is None else f"{row[key]:.4f}"
                      for key in ("precision", "recall", "map50")]
            lines.append("| " + row["class"] + " | " + " | ".join(values) + " |")
        lines.append("")
    speed_path = ROOT / "results/speed-evaluation.json"
    if speed_path.is_file():
        data = json.loads(speed_path.read_text(encoding="utf-8"))
        lines += ["## Full pipeline speed", "",
                  f"Video: `{data['video']}` ({data['video_seconds_used']:.1f} s)",
                  f"Frames processed: {data['frames_processed']} ({data['warming_frames']} warming)",
                  f"End-to-end processing FPS: {data['end_to_end_fps']:.3f}",
                  f"Risk-decision latency mean / p95: {data['latency_mean_ms']:.2f} / {data['latency_p95_ms']:.2f} ms",
                  f"Measured on: {data['machine']} ({data['device']})", ""]
    else:
        lines += ["## Full pipeline speed", "", "Not measured: needs a recorded video and zone polygon.", ""]
    scenario_path = ROOT / "results/scenario-evaluation.json"
    if scenario_path.is_file():
        data = json.loads(scenario_path.read_text(encoding="utf-8"))
        agreement = "N/A (expert labels are missing)" if data["band_agreement"] is None else f"{data['band_agreement']:.1%}"
        lines += ["## Expert scenarios", "", f"Scenarios evaluated: {data['scenario_count']}",
                  f"Expert band agreement: {agreement}",
                  f"Fuzzy-policy latency median / p95: {data['latency_median_ms']:.3f} / {data['latency_p95_ms']:.3f} ms",
                  ""]
    else:
        lines += ["## Expert scenarios", "", "Not measured: needs expert-reviewed scenario rows.", ""]
    (ROOT / "results/final_numbers.md").write_text("\n".join(lines), encoding="utf-8")


def run_maps(device):
    tasks = (("results/ppe_eval.json", "src/train_report.py"),
             ("results/fire_eval.json", "src/fire_report.py"))
    for output, script in tasks:
        target = ROOT / output
        if not target.is_file():
            model = ROOT / ("models/ppe_best.pt" if "ppe" in output else "models/fire_best.pt")
            if not model.is_file():
                raise FileNotFoundError(f"Trained model is missing: {model}")
            subprocess.run([sys.executable, str(ROOT / script), "--device", device],
                           cwd=ROOT, check=True)
    write_final_numbers("detector validation; see per-model machine labels")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_scenarios(path):
    from risk import compute_risk, get_engine
    if not path.is_file():
        raise FileNotFoundError(f"Scenario CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = set(SCENARIO_FIELDS) - {"scenario_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Scenario CSV missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Scenario CSV has headers but no scenarios; do not invent expert labels")
    get_engine()  # exclude one-time fuzzy-system construction from per-row latency
    outputs, latencies, seen = [], [], set()
    labelled = 0
    agree = 0
    for number, row in enumerate(rows, 2):
        scenario_id = (row.get("scenario_id") or f"row-{number - 1:03d}").strip()
        if not scenario_id or scenario_id in seen:
            raise ValueError(f"Row {number}: scenario_id must be nonempty and unique")
        seen.add(scenario_id)
        try:
            values = {key: float(row[key]) for key in SCENARIO_FIELDS[1:6]}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {number}: all five inputs must be numeric") from error
        tick = time.perf_counter_ns()
        score, band = compute_risk(**{key: values[key] for key in SCENARIO_FIELDS[1:5]},
                                   unhelmeted_in_zone_seconds=values["unhelmeted_in_zone_seconds"])
        latency_ms = (time.perf_counter_ns() - tick) / 1_000_000
        latencies.append(latency_ms)
        expected = (row.get("expected_band") or "").strip().title()
        if expected and expected not in BANDS:
            raise ValueError(f"Row {number}: expected_band must be one of {', '.join(BANDS)} or blank")
        match = None if not expected else band == expected
        if expected:
            labelled += 1
            agree += bool(match)
        outputs.append({"scenario_id": scenario_id, "score": score, "predicted_band": band,
                        "expected_band": expected or "", "match": "" if match is None else str(match).lower(),
                        "latency_ms": latency_ms})
    out_csv = ROOT / "results/scenario_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario_id", "score", "predicted_band",
            "expected_band", "match", "latency_ms"))
        writer.writeheader()
        writer.writerows(outputs)
    result = {"date": measured_at(), "input": str(path.resolve()),
              "input_sha256": sha256(path),
              "scenario_count": len(rows), "expert_label_count": labelled,
              "band_agreement": agree / labelled if labelled else None,
              "latency_median_ms": statistics.median(latencies),
              "latency_p95_ms": percentile(latencies, .95),
              "latency_scope": "fuzzy-policy calculation only; excludes detector inference",
              "results_csv": str(out_csv)}
    (ROOT / "results/scenario-evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_final_numbers("fuzzy scenarios only; per-model hardware appears in detector reports")


def run_speed(video, zones, seconds, machine, device):
    import cv2
    from app import SafetyPipeline, Settings
    for path in (video, zones, ROOT / "models/ppe_best.pt", ROOT / "models/fire_best.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"Required speed-test input is missing: {path}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not math.isfinite(fps_source) or fps_source <= 0 or frame_count <= 0:
        raise ValueError("Video must report a valid frame rate and frame count")
    duration = frame_count / fps_source
    if duration < seconds:
        raise ValueError(f"Need at least {seconds:g} seconds of video; this file is {duration:.2f} seconds")
    needed_frames = min(frame_count, math.ceil(seconds * fps_source))
    samples, timestamps = [], []
    ready_at = []
    def record(sample):
        now = time.perf_counter()
        if sample["status"] == "pipeline_ready":
            ready_at.append(now)
            return
        samples.append(sample)
        timestamps.append(now)
    settings = Settings(source=str(video), ppe_weights=ROOT / "models/ppe_best.pt",
                        fire_weights=ROOT / "models/fire_best.pt", zones=zones,
                        device=device, silent=True,
                        output_dir=ROOT / "results" / f"speed-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        max_frames=needed_frames)
    pipeline = SafetyPipeline(settings, frame_callback=record)
    pipeline.start()
    pipeline.thread.join()
    final_state = pipeline.snapshot()
    pipeline.stop()
    if final_state.get("error"):
        raise RuntimeError(f"Pipeline stopped with an error: {final_state['error']}")
    active = [row for row in samples if row["latency_ms"] is not None]
    if not active or len(samples) < needed_frames or not ready_at or timestamps[-1] <= ready_at[0]:
        raise RuntimeError(f"Only {len(samples)} of {needed_frames} frames reached the pipeline")
    latencies = [row["latency_ms"] for row in active]
    elapsed = timestamps[-1] - ready_at[0]
    result = {"date": measured_at(), "video": str(video.resolve()),
              "video_sha256": sha256(video),
              "video_seconds_used": len(samples) / fps_source,
              "source_fps": fps_source, "frames_processed": len(samples),
              "active_risk_decisions": len(active),
              "warming_frames": len(samples) - len(active),
              "processing_wall_seconds": elapsed,
              "end_to_end_fps": len(samples) / elapsed,
              "latency_mean_ms": statistics.mean(latencies),
              "latency_p95_ms": percentile(latencies, .95),
              "machine": machine, "device": device, "zone_config": str(zones.resolve()),
              "scope": "Full PPE/fire/zone/fuzzy application pipeline on recorded video; no physical camera latency"}
    (ROOT / "results/speed-evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_final_numbers(machine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("map", "speed", "scenarios", "all"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--zones", type=Path)
    parser.add_argument("--scenario-file", type=Path, default=ROOT / "scenarios/scenarios.csv")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--machine", default="Laptop; specify exact CPU/GPU if known")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.seconds):
        parser.error("--seconds must be positive and finite")
    try:
        if args.map or args.all:
            run_maps(args.device)
        if args.speed or args.all:
            if args.video is None or args.zones is None:
                parser.error("--speed and --all require --video and --zones")
            run_speed(args.video.resolve(), args.zones.resolve(), args.seconds, args.machine, args.device)
        if args.scenarios or args.all:
            run_scenarios(args.scenario_file.resolve())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
