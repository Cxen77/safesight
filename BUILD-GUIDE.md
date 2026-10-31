# BUILD GUIDE — CIS-HW-01: Edge AI Safety Monitoring on Jetson Nano
## Everything that can be finished BEFORE the hardware arrives

**The idea in one line:** one camera → two small AI models (PPE + fire) → zone geometry → a fuzzy-logic brain that outputs a Safety Risk Score (Low / Medium / High / Critical) → buzzer, dashboard, event log — all measured for accuracy, latency and FPS.

**The good news:** ~90% of this project is software. Every phase in this document runs on a normal laptop. Only the last section (Phase 9) touches the Jetson Nano, and it is written as a checklist to execute the day the board arrives.

**How to use this doc**
- Do the phases **in order**. Each phase ends with a "Done when" test — don't move on until it passes.
- Every phase has a **📋 PROMPT** block. Copy-paste it into your AI coding tool (Claude / ChatGPT / Cursor) exactly as written, then run what it gives you. The prompts already contain every number and rule from the research, so don't edit them unless a step tells you to fill in a value.
- **One sacred rule:** the test split (Phase 2) is frozen forever. Never train on test images, never tune on the staged demo scenes. A published fire-detection paper's famous accuracy "gain" turned out to be test data leaking into training — we do not repeat that.

**Based on (credit these in your README — the base repo is GPL-3.0, so keep your project open-source too):**
- Base project: https://github.com/Tanishjain-glitch/PPE-Detection-Jetson (YOLOv8n PPE weights + TensorRT path)
- Dataset + pretrained weights: https://github.com/ahmadmughees/SH17dataset (SH17, CC BY-NC-SA 4.0)
- Fire dataset: https://github.com/gaia-solutions-on-demand/DFireDataset (read its LICENSE first)
- Classic-Nano runtime (if faculty board is the old Nano): https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano
- Research: ~/workspace/wop-ps/jetson-papers.md · Full plan: ~/workspace/wop-ps/PLAN-jetson-safety.md

**Folder structure (create exactly this in Phase 0):**
```
jetson-safety/
  models/          # trained .pt weights live here (ppe_best.pt, fire_best.pt)
  data/
    sh17/          # downloaded SH17 subset
    dfire/         # downloaded D-Fire
    own/           # YOUR lab photos (raw, then annotated export)
    dataset_ppe/   # merged training dataset (YOLO format) + data.yaml
    dataset_fire/  # fire training dataset + data.yaml
    splits_frozen/ # the frozen test split — DO NOT TOUCH after Phase 2
  src/
    infer.py       # Phase 1: camera + model demo
    dataset_tools.py  # Phase 2: merge/verify/split
    zones.py       # Phase 5: polygon zones, intrusion, proximity
    risk.py        # Phase 6: fuzzy Safety Risk Score
    alerts.py      # Phase 7: buzzer/LED driver (mock on laptop)
    app.py         # Phase 7: Flask dashboard + event log
    evaluate.py    # Phase 8: mAP / latency / FPS / scenario tests
  scenarios/       # Phase 8: the 30 scripted validation cases (CSV)
  exports/         # Phase 9: ONNX exports, ready for the board
  results/         # every measurement table lands here as CSV/MD
```

---

## PHASE 0 — Setup (Day 1, ~1 hour)

**Step 0.1 — Install.** Python 3.10 or 3.11, Git, VS Code. Then in a terminal inside a new `jetson-safety` folder:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/Mac: source .venv/bin/activate
pip install ultralytics opencv-python numpy flask scikit-fuzzy control pyyaml
```

**Step 0.2 — Download the open-source base.**
```bash
git clone https://github.com/Tanishjain-glitch/PPE-Detection-Jetson base-ppe
git clone https://github.com/ahmadmughees/SH17dataset base-sh17
git clone https://github.com/gaia-solutions-on-demand/DFireDataset data/dfire
```
From `base-sh17`, also download the **pretrained YOLOv8 weights from its GitHub Releases** page and put the nano one in `models/` as `sh17_yolov8n.pt`. From `base-ppe`, copy its `best.pt` into `models/` as `ppe_base_best.pt`.

📋 **PROMPT 0 — scaffold**
```
Create the folder structure for a project called jetson-safety exactly as I describe: folders models/, data/sh17, data/dfire, data/own, data/dataset_ppe, data/dataset_fire, data/splits_frozen, src/, scenarios/, exports/, results/. Create empty Python files in src/: infer.py, dataset_tools.py, zones.py, risk.py, alerts.py, app.py, evaluate.py. Create a requirements.txt with: ultralytics, opencv-python, numpy, flask, scikit-fuzzy, control, pyyaml. Create a README.md with the project title "Edge AI Safety Monitoring on Jetson Nano (CIS-HW-01)" and a Credits section listing: base PPE project Tanishjain-glitch/PPE-Detection-Jetson (GPL-3.0), SH17 dataset by ahmadmughees (CC BY-NC-SA 4.0), D-Fire dataset. Do not write any code yet, only the structure.
```

**Done when:** folders exist, `pip list` shows ultralytics + scikit-fuzzy, both downloaded weights are in `models/`.

---

## PHASE 1 — See it work (Day 1–2)

Goal: a pretrained safety model running on your laptop webcam TODAY. This is also your pitch-week proof that the approach works.

📋 **PROMPT 1 — inference demo**
```
Write src/infer.py in Python using ultralytics YOLO and OpenCV. Requirements:
- Load a YOLO weights file given as a command-line argument (--weights), default models/sh17_yolov8n.pt.
- Input: webcam if --source 0, otherwise a video file path or image folder given by --source.
- For every frame: run detection at imgsz=640, draw bounding boxes with class name + confidence. Use these colors: person = white, helmet = green, head = red, vest/safety vest = yellow, all other classes = grey.
- Overlay in the top-left corner: current FPS (smoothed over the last 30 frames) and the number of people currently detected.
- Save the annotated output to results/demo_output.mp4 when the input is a video; show it live in a window for webcam; quit on 'q'.
- Print the model's class list at startup so I can verify the class names.
No other features. Keep it under 120 lines.
```

Run it: `python src/infer.py --weights models/sh17_yolov8n.pt --source 0` — walk in front of the webcam wearing any cap/helmet you have.

**Done when:** boxes track people on webcam at a smooth rate and you have one saved demo video in `results/`. (Screenshot this for the pitch.)

---

## PHASE 2 — Your own data: the accuracy weapon (Week 1–2)

The research is blunt: a model trained only on public photos dropped to **24% helmet recall** on a new site. Your lab photos are what fix that. This phase matters more than any code.

**Step 2.1 — Shoot the photos (2–3 sessions, phone camera is fine).** Target **400–600 photos** in the actual lab/room where the demo will happen, with the actual helmet and vest you'll use:
- Helmet ON / helmet OFF (bare head) / helmet carried in hand / helmet on a table
- Vest ON / vest OFF; half-zipped counts as on
- Distances: close (1–2 m), mid (3–4 m), far (5–6 m) — far shots matter, helmets get tiny
- Angles: front, side, back, slightly above (camera height)
- Lighting: normal, tube-light only, window glare
- 2+ people in frame, partial occlusion (person half behind a desk), sitting and standing
- At least 3 different people; get their verbal OK to be photographed for the project
- **Over-collect the weak classes:** helmet and vest shots should each be ≥150 photos. These are the classes public data starves.

**Step 2.2 — Annotate.** Upload to **Roboflow** (free tier) or use **CVAT**. Draw boxes with ONLY these class names (must match the SH17 subset you train with): `person`, `head`, `helmet`, `vest`. Rule: a person wearing a helmet gets THREE boxes (person, helmet — and no head box; the head box is only for a visible bare head). Export in **YOLO format** into `data/own/`.

**Step 2.3 — Merge + freeze the split.** Merge: SH17 images for classes person/head/helmet/vest (from `data/sh17`) + your Roboflow export. Split **70% train / 15% val / 15% test**, stratified by class. Copy the test portion into `data/splits_frozen/` and **never touch it again** — not for training, not for tuning, not for "just one more epoch check".

📋 **PROMPT 2 — dataset tools**
```
Write src/dataset_tools.py with three commands using argparse (Python stdlib + pyyaml only):
1) merge: inputs are two YOLO-format dataset folders (--a, --b), each with images/ and labels/ subfolders and a data.yaml. Output a merged dataset in --out with images/ and labels/, keeping only these classes: person, head, helmet, vest. Remap class ids in the label files to the new order: person=0, head=1, helmet=2, vest=3. If the two datasets use different class names for the same thing (e.g. "safety vest" vs "vest"), accept a --rename argument in the form old:new (can be given multiple times).
2) split: take a merged dataset folder (--data) and create train/val/test subfolders with a 70/15/15 split, stratified so each class appears in roughly the same proportion in every split (use the set of classes present in each image's label file). Use seed 42. Write a new data.yaml in the output root pointing at the splits with names: 0 person, 1 head, 2 helmet, 3 vest. Also copy the test split into a separate folder given by --frozen.
3) report: given a dataset folder with train/val/test splits, print a table: for each split, number of images and number of boxes per class, and warn in capital letters if any class has fewer than 50 boxes in train.
The script must never modify the folder given to --frozen.
```

**Done when:** `report` shows ≥50 train boxes for every class (aim much higher for helmet/vest), and `data/splits_frozen/` exists and is closed forever.

---

## PHASE 3 — Train the PPE model (Week 2)

Train on a GPU if you can: college lab GPU, or Google Colab free tier (upload the dataset folder, same commands). Starting point is the SH17 pretrained weights — transfer learning, NOT training from zero.

```bash
yolo detect train model=models/sh17_yolov8n.pt data=data/dataset_ppe/data.yaml imgsz=640 epochs=60 batch=16 patience=15 project=results name=ppe_train1
```
Copy the resulting `results/ppe_train1/weights/best.pt` to `models/ppe_best.pt`. Then evaluate honestly:

```bash
yolo detect val model=models/ppe_best.pt data=data/dataset_ppe/data.yaml split=test imgsz=640
```

📋 **PROMPT 3 — training report**
```
Write a Python script src/train_report.py that: takes a YOLO weights file and a data.yaml as arguments; runs validation on the test split using ultralytics; writes results/ppe_eval.md containing: overall [email protected] and [email protected]:0.95, a per-class table (class, precision, recall, [email protected]), the confusion matrix saved as results/confusion_matrix.png, and a plain-English section "Weak spots" that lists every class with recall below 0.6 and says how many more training photos of that class I should collect (rule: if recall < 0.5 suggest 150 more, if 0.5–0.6 suggest 75 more). Use the model's own validation API, do not parse console text.
```

**Reality check (from the papers):** overall test [email protected] in the **0.55–0.65** band is the honest, publishable range for a nano model. If helmet recall is the weak one, the fix is more helmet photos (Phase 2), not more epochs.

**Done when:** `models/ppe_best.pt` exists and `results/ppe_eval.md` shows the per-class table with no class catastrophically low (recall ≥ ~0.5 after one data top-up round).

---

## PHASE 4 — Train the fire/smoke model (Week 3)

Same recipe, different dataset. D-Fire is already YOLO format with classes `fire` and `smoke`. Make its data.yaml (train/val/test 70/15/15, seed 42 — reuse Prompt 2's split logic or do it by hand), and add 20–40 of your own staged clips later in Phase 7 testing (incense stick, candle — filmed, never near anything flammable, faculty/lab permission first).

```bash
yolo detect train model=yolov8n.pt data=data/dataset_fire/data.yaml imgsz=640 epochs=50 batch=16 patience=15 project=results name=fire_train1
```
Copy best weights to `models/fire_best.pt`. **Expected honest result: [email protected] ≈ 0.70–0.76, smoke easier than flame.** In the final report, fire is an *alert layer*, never a "certified fire detector" — say so in writing before a judge says it for you.

📋 **PROMPT 4 — fire model notes**
```
Write results/fire_eval.md for me as a filled template after I paste you the validation output of my fire model: sections are (1) per-class table for fire and smoke with precision/recall/[email protected], (2) a paragraph explaining in simple English why smoke usually scores higher than flame in published D-Fire results, (3) a "Claim we can make" line that states my measured [email protected] and frames the model as an early-warning alert layer, and (4) a "Claim we must NOT make" line. Wait for me to paste the validation numbers before writing any values.
```

**Done when:** `models/fire_best.pt` exists and detects a candle/incense test video (record one on your phone) at least sometimes at confidence ≥0.4.

---

## PHASE 5 — Zone geometry: the part that needs no AI (Week 3)

The camera is fixed, so the danger zone is just a shape on the screen. Draw it once, save it, reuse forever.

**Concepts (simple):** for each detected person box, take the **bottom-centre point** (their "feet"). Feet inside the polygon = intrusion. A timer per person ID (use the tracker: `model.track()` gives stable IDs) = intrusion duration. Crowd = count of feet inside. Proximity = shortest distance from feet point to the polygon edge, converted to a 0–1 margin (1 = far/safe, 0 = touching the line).

📋 **PROMPT 5 — zones module**
```
Write src/zones.py in Python with OpenCV and numpy:
1) A function draw_zone(image_path_or_frame): opens the frame in a window; the user clicks points to draw a polygon (left click adds a point, right click removes the last point, ENTER closes and saves, ESC cancels). On save, write the points (as fractions of frame width/height, so they survive resolution changes) to zones.json next to the script, and also save an annotated preview image to results/zone_preview.png.
2) A class ZoneMonitor: constructed with the frame size and the zones.json path. Method update(person_boxes_with_ids, frame_shape) where each item is (track_id, x1, y1, x2, y2). For each person compute the feet point (bottom-centre). Return a dict with: intruders (list of track_ids inside the polygon), duration_s (dict track_id -> seconds continuously inside; reset when they leave), crowd_count (people inside), proximity_margin (0..1: for the closest intruder, 1 minus their normalised distance to the polygon boundary, where the normaliser is 15% of frame width; people outside get margin based on distance to the polygon, min 0).
Use cv2.pointPolygonTest for inside tests and distances. Keep a small internal state dict for durations using time.monotonic. Include a __main__ demo that runs it on a webcam feed with models/ppe_best.pt and prints the dict once per second.
```

**Done when:** standing in the taped zone on your floor makes `duration_s` climb, stepping out resets it, and a second person raises `crowd_count` to 2.

---

## PHASE 6 — The fuzzy brain: Safety Risk Score (Week 4) ⭐ the graded centrepiece

This is the PS's signature requirement — the part most teams will fake with three if-statements. Yours is a real Mamdani fuzzy system, and every rule can be read aloud to a judge.

**Inputs (each 0–1, each with terms low / medium / high):**
| Input | Meaning | How it's computed |
|---|---|---|
| `ppe_load` | How bad the PPE violations are | (unhelmeted heads + unvested persons) ÷ max(persons,1); a bare head counts double vs a missing vest |
| `zone_pressure` | Intrusion weight | combines crowd_count (0 / 1 / 2+) and longest duration (0s / <5s / ≥5s) into one 0–1 value |
| `proximity` | How close to the hazard line | from Phase 5's proximity_margin, inverted (close = high) |
| `fire_evidence` | Fire model belief | max fire/smoke confidence this frame, smoothed over the last 10 fire-checks |

**Structure — two stages** (keeps rules few and auditable):
- **Stage 1:** ppe_load + zone_pressure + proximity → `hazard` (low/medium/high). ~15 rules.
- **Stage 2:** hazard + fire_evidence → `risk` 0–100. ~9 rules.
- **Bands:** 0–25 Low · 26–50 Medium · 51–75 High · 76–100 Critical.
- **Hard overrides (checked before fuzzy):** fire_evidence ≥ 0.7 → Critical. Person inside zone with bare head for ≥ 3 s → at least High; ≥ 8 s → Critical.

**Core rules (plain English — Prompt 6 turns these into code):**
Stage 1: IF ppe high AND zone high THEN hazard high · IF ppe high AND zone medium THEN hazard high · IF ppe medium AND zone high THEN hazard high · IF ppe low AND zone low AND proximity low THEN hazard low · IF proximity high AND zone medium-or-high THEN hazard high · IF all medium THEN hazard medium … (the prompt asks the AI to complete the full consistent table.)
Stage 2: IF hazard high AND fire medium-or-high THEN risk Critical-range · IF hazard high AND fire low THEN risk High-range · IF hazard medium AND fire high THEN risk High-range · IF hazard low AND fire low THEN risk Low-range · IF hazard medium AND fire low THEN risk Medium-range …

📋 **PROMPT 6 — fuzzy risk engine**
```
Write src/risk.py in Python using scikit-fuzzy (skfuzzy + skfuzzy.control). Implement a two-stage Mamdani fuzzy system:
STAGE 1 inputs (universe 0..1, triangular membership functions for low/medium/high): ppe_load, zone_pressure, proximity. Output: hazard (universe 0..1, terms low/medium/high).
STAGE 2 inputs: hazard (from stage 1) and fire_evidence (0..1, terms low/medium/high). Output: risk (universe 0..100, terms low/medium/high/critical with triangular membership functions centred near 12, 38, 63, 88).
Write a COMPLETE and consistent rule base: stage 1 must cover all 27 combinations of the three inputs (use AND = min). The logic must be monotonic: raising any input must never lower hazard. General principle: hazard is high if any two inputs are high, or ppe_load is high while zone_pressure is at least medium, or proximity is high while zone_pressure is at least medium; hazard is low only when all inputs are low or at most one is medium with the rest low; everything else is medium.
Stage 2 rules (9, all combinations): risk follows the higher of hazard and fire_evidence, pushed one band up when both are at least medium; specifically: hazard high + fire high = critical; hazard high + fire medium = critical; hazard high + fire low = high; hazard medium + fire high = high; hazard medium + fire medium = high; hazard medium + fire low = medium; hazard low + fire high = high; hazard low + fire medium = medium; hazard low + fire low = low.
Before the fuzzy computation, apply hard overrides in a wrapper function compute_risk(ppe_load, zone_pressure, proximity, fire_evidence, unhelmeted_in_zone_seconds): if fire_evidence >= 0.7 return (100, "Critical"); if unhelmeted_in_zone_seconds >= 8 return at least (85, "Critical"); if >= 3 return at least (65, "High") — "at least" means take the max of the fuzzy result and the override floor.
compute_risk returns (score_float, band_string) where bands are: 0-25 Low, 26-50 Medium, 51-75 High, 76-100 Critical.
Also write a __main__ self-test that prints compute_risk for these cases and their expected bands: (0,0,0,0,0)=Low; (0.9,0.9,0.5,0,0)=High or Critical; (0,0,0,0.8,0)=Critical; (0.5,0.5,0.5,0,4)=at least High; (0.2,0.1,0.1,0,0)=Low.
Use centroid defuzzification. Keep the rule definitions readable with comments in plain English.
```

**Done when:** the self-test prints the expected bands, and wiggling any input upward never lowers the score (try 3 random cases by hand).

---

## PHASE 7 — Dashboard, alerts, event log (Week 4–5)

**Alert behaviour (decide now, code once):** a violation must persist **1 second** before it logs (kills flicker). High/Critical turns the buzzer on; Critical **latches** — it stays until the scene is clear for 3 continuous seconds. Every High/Critical event saves a snapshot photo + a log row (time, band, score, cause inputs). On the laptop, the "buzzer" is a beep + red screen banner; the real GPIO driver is written in the same file behind a switch, ready for the board.

📋 **PROMPT 7 — dashboard app**
```
Write src/app.py: a Flask dashboard for the safety system, plus src/alerts.py.
alerts.py: a class AlertDriver with method set_level(band) where band is Low/Medium/High/Critical. On laptop mode (constructor arg hardware=False) it plays a beep using winsound on Windows or prints "BUZZER ON" on other systems, only when band is High or Critical, and prints every band change. With hardware=True it should try to import Jetson.GPIO and drive pin 12 (buzzer) and pin 16 (LED) — write this code path carefully but it is OK if it is untested on the laptop. Switching bands must not spam: only act on changes.
app.py requirements:
- Runs the full pipeline in a background thread: webcam/video source, PPE model every frame (models/ppe_best.pt with tracking), fire model (models/fire_best.pt) every 10th frame, ZoneMonitor from src/zones.py, risk from src/risk.py.
- Computes ppe_load, zone_pressure, proximity, fire_evidence each frame from the detections as specified in the project design (I will paste the definitions if you need them — ask me if any input is unclear rather than inventing a formula).
- Smoothing: keep the max fire confidence over the last 10 fire checks as fire_evidence.
- Debounce + latch: a High or Critical band must persist 1 second before an event is logged; Critical stays latched until 3 continuous seconds below High.
- Event log: append to results/events.csv with columns timestamp, band, score, ppe_load, zone_pressure, proximity, fire_evidence, cause (a short string like "unhelmeted person in zone 6s") and save a snapshot image to results/snapshots/<timestamp>.jpg for every logged event.
- Web page at / : live MJPEG video stream with boxes and zone polygon drawn, a big risk gauge (score number + band, background color green/yellow/orange/red), current FPS and latency in ms, and a table of the last 10 events (auto-refresh). Keep the HTML simple, in one template string.
- A /api/state route returning JSON with score, band, fps, latency_ms, and the four input values.
Do not simplify away the debounce, latch, or snapshot behaviour.
```

**Done when:** you can run the whole demo on the laptop — walk into the zone bare-headed, watch the gauge climb, hear the beep, and find your photo in `results/snapshots/`.

---

## PHASE 8 — Measure everything (Week 5) 📊 the PS explicitly grades this

Three numbers + one validation. All produced by one script so the report writes itself.

📋 **PROMPT 8 — evaluation harness**
```
Write src/evaluate.py with four modes (argparse):
1) --map: run ultralytics validation of models/ppe_best.pt and models/fire_best.pt on their frozen test splits; write results/final_numbers.md with per-model overall [email protected] and per-class precision/recall/[email protected] tables.
2) --speed: run the full pipeline (both models time-sliced, zones, fuzzy) on a 60-second recorded test video (--video) at 640x640; measure and append to results/final_numbers.md: end-to-end FPS (frames fully processed per second), average and p95 latency in ms from frame capture to risk decision, and state clearly these were measured on THIS machine (print the machine description I pass via --machine, e.g. "laptop RTX 3050" or "Jetson Nano classic").
3) --scenarios: read scenarios/scenarios.csv with columns ppe_load, zone_pressure, proximity, fire_evidence, unhelmeted_in_zone_seconds, expected_band; run src/risk.py compute_risk on each row; write results/scenario_results.csv with the system's band next to the expected band and print the agreement percentage.
4) --all: run 1-3 in order.
The script must label every number with the machine and date. No number may be typed by hand anywhere — everything computed.
```

**The 30 scenarios** (create `scenarios/scenarios.csv`): you + one faculty member or senior act as the "expert": invent 30 situations (empty room; one visitor at the door with helmet; worker inside zone no helmet 2s / 5s / 10s; two people crowded; smoke only, nobody present; person near the line but outside; …), fill the four inputs for each, and write the band a safety officer would choose. Target: the literature band for such systems is **85–90% agreement** — if you land there, that's a publishable sentence in your report.

**Done when:** `results/final_numbers.md` exists with mAP tables, a labelled FPS/latency line, and scenario agreement ≥ ~80% (tune fuzzy membership edges — NOT the test data — if below).

---

## PHASE 9 — Board day checklist (do the prep NOW, run it when the Nano arrives)

**Prep you can finish today (laptop):**
```bash
yolo export model=models/ppe_best.pt format=onnx imgsz=640
yolo export model=models/fire_best.pt format=onnx imgsz=640
```
Move both `.onnx` files to `exports/`. Also write down your GPIO plan: buzzer → pin 12, LED → pin 16, ground shared.

**Ask faculty (at the pitch, before anything else):**
1. "Is the board the **classic Nano (JetPack 4.6)** or the **Orin Nano (JetPack 5/6)**?" — this one answer picks the runtime path below.
2. "When do we receive it, and can we keep it for the full 8 weeks?"
3. "Can we use a lab monitor/keyboard, or should we run it headless over SSH?"

**On the board, in order:**
1. Power it in **10 W mode** (`sudo nvpmodel -m 0`), check JetPack: `cat /etc/nv_tegra_release`.
2. Camera check: `ls /dev/video*` and a test capture — before any AI work.
3. Convert models to TensorRT FP16:
   - **Orin path:** follow the base repo's DeepStream/TensorRT steps.
   - **Classic path:** use the Qengineering repo's TensorRT-8 conversion for the ONNX files.
4. Point `app.py` at the TensorRT engines (same pipeline, engine loader instead of `.pt`), set `AlertDriver(hardware=True)`, wire buzzer/LED, re-run Phase 8 `--speed --machine "Jetson Nano"` and `--scenarios` on the board.
5. **Expected honest board numbers:** 10–15 FPS end-to-end at 640 (classic) — if you land there, you are exactly on the published range; put the Qengineering table and your table side by side in the report.
6. Autostart on boot (systemd service) only after everything else passes — a demo that starts itself survives demo-day nerves.

📋 **PROMPT 9 — porting help (use on board day)**
```
I have a working Python safety-monitoring pipeline on my laptop using ultralytics YOLO .pt weights. I am moving it to a Jetson Nano [PASTE: classic JetPack 4.6 OR Orin JetPack 6 — fill this in]. I have ONNX exports of both models. Give me: (1) the exact steps and commands to convert both ONNX models to TensorRT FP16 engines on this board, (2) the minimal code change to load and run a TensorRT engine in place of model = YOLO("file.pt") while keeping identical pre/post-processing (letterbox 640, NMS thresholds), and (3) a checklist of the three most common reasons TensorRT conversion fails on this JetPack version and how to fix each. Do not rewrite my whole pipeline; only the model-loading part changes.
```

---

## What "pre-hardware complete" means (your checklist)
- [ ] Phase 1 demo video saved (pitch proof)
- [ ] Own dataset shot, annotated, split frozen
- [ ] `ppe_best.pt` + `ppe_eval.md` (honest per-class numbers)
- [ ] `fire_best.pt` (candle test passes)
- [ ] Zones working on webcam (intrusion timer + crowd)
- [ ] Fuzzy self-test passes; overrides verified
- [ ] Full laptop demo runs end-to-end with buzzer-beep, dashboard, snapshots, event log
- [ ] `final_numbers.md`: mAP + FPS/latency (labelled "laptop") + scenario agreement
- [ ] Both models exported to ONNX in `exports/`
- [ ] Pitch answers ready: which Nano, when, monitor or headless

When every box is ticked, the board becomes a 2–3 day porting job (Phase 9) instead of an 8-week risk. That is the whole strategy: **finish the project before the hardware arrives; let the hardware be the last step, not the first.**
