# Edge AI Safety Monitoring on Jetson Nano (CIS-HW-01)

Fixed-camera PPE and fire/smoke detection, zone monitoring, and a two-stage
Mamdani fuzzy Safety Risk Score. Laptop development precedes Jetson deployment.

Follow [the build guide](BUILD-GUIDE.md) in phase order, with the documented
public-only dataset adjustment.
Implementation status is tracked in [PROGRESS.md](PROGRESS.md).

## Setup (Windows PowerShell)

Use Python 3.11 (or 3.10 as specified in the guide).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Calling the environment's Python directly avoids PowerShell activation-policy issues.
The verified environment is captured in `requirements-lock.txt`; use
`-r requirements-lock.txt` to reinstall those exact package versions.

## Project layout

- `models/`: downloaded and trained weights.
- `data/sh17/`, `data/dfire/`, `data/own/`: source data.
- `data/dataset_ppe/`, `data/dataset_fire/`: prepared training datasets.
- `data/splits_frozen/`: reserved for immutable test splits in Phase 2.
- `src/`: inference, dataset preparation, zones, risk, alerts, dashboard, evaluation.
- `scenarios/`: expert-labelled validation cases.
- `exports/`: ONNX and board deployment artifacts.
- `results/`: measured reports, demo video, snapshots, and event logs.

Do not train on the frozen test split or tune on staged demo scenes.
Do not report accuracy, FPS, or latency until measured. Fire detection is an
experimental early-warning alert layer.

## Run the Phase 1 demo

```powershell
cd path\to\safesight
.\.venv\Scripts\python.exe src/infer.py --source 0
```

Stand in view, try a real helmet if available, then press **q** in the video
window to stop. A cap can test visibility but is not safety equipment and may
not be detected as a helmet. Webcam and video input save annotated footage to
`results/demo_output.mp4`; subsequent runs replace this output. Use
`--output results/my_demo.mp4` to keep a separate recording. Camera index 1 may
be needed for an external webcam. Close other apps using the camera if capture fails.

For a recorded clip or folder of images:

```powershell
.\.venv\Scripts\python.exe src/infer.py --source "C:\path\clip.mp4"
.\.venv\Scripts\python.exe src/infer.py --source "C:\path\images" --no-display
```

Images are saved under `results/demo_images/`. `--max-frames 30 --no-display`
is available for bounded checks without opening a window. Detection does not
assign persistent person IDs at this phase; tracking is added with zones later.

## Training data

The webcam demo uses existing weights and needs no training dataset download.
For training we need these three sources:

| Source | Purpose | Local destination |
| --- | --- | --- |
| [SH17](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection) | PPE images + YOLO labels; retain person, head, helmet, safety-vest | `data/sh17/` |
| [D-Fire mirror linked by its authors](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) | Fire/smoke images + YOLO labels, including negative images | `data/dfire/dataset/` |
| Your own photos (optional) | Later adaptation to the camera/site if available | `data/own/` |

The downloader retrieves the public SH17 (~14.1 GB) and D-Fire (~3.0 GB)
archives, resumes interrupted transfers, validates ZIP entries while extracting,
and retains the archives in `data/downloads/`. JSON progress files in that
folder show bytes downloaded and state (downloading/extracting/complete/failed).
The source repository in `data/dfire/` is preserved separately from its images.

If the existing downloader has stopped, rerun from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts/download_datasets.py
```

Do not launch multiple copies. Raw datasets stay untouched by training preparation;
we will inspect their label maps and source splits before making Phase 2 splits.

If site photos become available later, aim to collect 400–600 photos with permission from at least three
participants. Vary helmet/vest on and off, distance, angle, lighting, occlusion,
and the number of people. Include helmets carried or placed on tables, and aim
for at least 150 photos containing helmets and 150 containing vests. Annotate
with only `person`, `head`, `helmet`, `vest` in CVAT or Roboflow and export YOLO
images + labels + class-name mapping. Use `head` for visible bare heads, as the
guide specifies. Keep raw photos as well as the annotated export.

The current baseline uses public data only because site photos are unavailable.
Training and test splitting are not performed by the downloader.

## Phase 2 preparation tools

The prepared public dataset is `data/ppe_public/`. All 8,099 image/label pairs
were validated, and exact image hashes checked for duplicates. Labels retain
only person=0, head=1, helmet=2, vest=3. Images are hardlinked to save disk space:
**do not modify either source or prepared images in place**. Use copies for
image augmentation or editing. YOLO's normal read-only training input is fine.

```powershell
.\.venv\Scripts\python.exe src/dataset_tools.py report --data data/ppe_public
.\.venv\Scripts\python.exe scripts/preview_ppe_split.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The preview script targets a 70/15/15 multilabel split with seed 42; rerunning
that script only writes a report. The actual public baseline splits have now
also been created separately as described below. D-Fire's existing split structure is preserved;
26 invalid label files are listed in `results/dfire-label-issues.json` for review
before fire-model training.

Merge refuses conflicting labels for identical images and deduplicates matching
ones. Label files must exist, including empty files for negative images. Outputs
must be empty; no overwrite option is provided. Split also refuses any existing
nonempty frozen destination before writing outputs. Later site data can be added
to development data without replacing or re-splitting the frozen baseline test set.

## Public-data baseline (current run)

The user confirmed that site photos are unavailable and the demo surroundings
may change. Training therefore proceeds using public SH17 data only:

| Split | Images | Purpose |
| --- | ---: | --- |
| Train | 5,669 | Weight updates |
| Validation | 1,214 | Early stopping and checkpoint selection |
| Test | 1,216 | Reserved evaluation after training |

Prepared data lives in `data/dataset_ppe/`; the independent frozen snapshot is
`data/splits_frozen/ppe/`. Seed is 42 and class order is person/head/helmet/vest.
Exact image-byte overlap across splits is zero. Visual near-duplicates and
cross-dataset pretraining overlap have not been audited.

The starting model is the general **COCO `models/yolov8n.pt`**, rather than the
SH17 PPE checkpoint that was already trained using SH17. This avoids known
SH17-specific pretraining exposure of the baseline test split. It does not
establish accuracy at an unseen demo location; check new surroundings separately.

Training uses `train_only.yaml` (train + validation, no test entry). The evaluation
descriptor `eval_frozen.yaml` uses the test working copy outside the frozen
folder, so YOLO cache files cannot modify the snapshot. Before test evaluation,
run `scripts/verify_ppe_split.py` to check image and label hashes in both copies.
Do not tune on frozen test results.

The public baseline started on an RTX 4060 with settings of 640 pixels, batch 8,
up to 60 epochs, patience 15, two data workers, and no RAM image cache. The run
was interrupted after three completed epochs; it has not been evaluated on the
reserved test split. Windows training uses a main guard. Reference:
[Ultralytics training](https://docs.ultralytics.com/modes/train/) and
[PyTorch GPU setup](https://pytorch.org/get-started/locally/).

Progress files (do not start a duplicate controller):

- `results/baseline-setup.json`: GPU installation / training stage.
- `results/public-baseline.log` and `results/public-baseline-errors.log`: controller output.
- `results/ppe-training-status.json`: epoch and validation progress once training starts.
- `results/ppe_public_baseline/weights/`: local training checkpoints.
- `checkpoints/ppe-baseline-resume.pt`: checkpoint saved after epoch 3, backed
  up in this repository so the run can be resumed after local cleanup.
- `models/ppe_best.pt`: copied only after a successful completed run.

If interrupted after a checkpoint is saved, resume from the project directory:

```powershell
.\.venv\Scripts\python.exe src/train.py --resume results/ppe_public_baseline/weights/last.pt
```

Resume only after checking that the previous process has stopped. Final test
evaluation is a separate step; no accuracy result is available during setup.

### Restore after clearing local files

The datasets and prepared splits are intentionally not stored in GitHub. The
SH17 and D-Fire archives total about 17 GB and can be fetched again with the
download script. After cloning this repository, recreate the PPE training split
with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe scripts/download_datasets.py --dataset sh17
.\.venv\Scripts\python.exe src/dataset_tools.py merge --a data/sh17 --a-yaml config/sh17-classes.yaml --out data/ppe_public --rename safety-vest:vest --link-images
.\.venv\Scripts\python.exe src/dataset_tools.py split --data data/ppe_public --out data/dataset_ppe --frozen data/splits_frozen/ppe --link-images
.\.venv\Scripts\python.exe scripts/download_datasets.py --dataset dfire
.\.venv\Scripts\python.exe src/prepare_fire.py --source data/dfire/dataset --out data/dataset_fire --frozen data/splits_frozen/fire --link-images
New-Item -ItemType Directory -Force results/ppe_public_baseline/weights
Copy-Item checkpoints/ppe-baseline-resume.pt results/ppe_public_baseline/weights/last.pt
.\.venv\Scripts\python.exe src/train.py --resume results/ppe_public_baseline/weights/last.pt
```

This checkpoint is a partial training resume point, not a finished or validated
model. The saved checkpoint reports three completed epochs. After training
finishes, the existing evaluator will validate the untouched frozen test split.
The original D-Fire data must also be downloaded and prepared before fire-model
training. Dataset availability and download URLs may change over time.

Fire training is queued to start only after the PPE run and PPE test evaluation
finish. It uses the same CUDA GPU, 640-pixel images, batch 8, up to 50 epochs,
and patience 15. Queue status is in `results/fire-queue-status.json`; fire
training status is in `results/fire-training-status.json`. It will verify the
frozen D-Fire split before producing `results/fire_eval.md` and
`results/fire_eval.json`. Do not start another GPU job while that queue or run is
active.

After both test reports exist, `scripts/publish_after_training.py` stages the
two final inference weights, measured reports, confusion matrices, and final
summary to GitHub. It will not publish optimizer checkpoints beyond the
resume checkpoint already included. SH17-derived weights retain the dataset's
CC BY-NC-SA 4.0 attribution and noncommercial/share-alike terms; see Credits.

The optional post-training evaluator script waits for successful completion,
verifies frozen test hashes, and calls `src/train_report.py` once. Start it after
training has started with `python scripts/evaluate_when_ready.py` if you want
automatic evaluation. A successful evaluation produces `results/ppe_eval.md`,
`results/ppe_eval.json`, and `results/confusion_matrix.png`, using measured
values from Ultralytics' API. It refuses to overwrite an existing test report.
If training fails or is interrupted, inspect the run before resuming.

## D-Fire preparation

`src/prepare_fire.py` prepares `data/dataset_fire/` from the downloaded source.
It preserves upstream split assignments, excludes entire image-label pairs with
invalid annotations, and checks SHA-256 image hashes. Duplicate priority is test,
then validation, then training, so a held-out image cannot remain in development
data. Negative images with valid empty label files are retained. Original files
are not edited. The declared class IDs remain **0 smoke, 1 fire**.

The test snapshot is independent in `data/splits_frozen/fire/`; images in the
prepared working dataset use hardlinks to save space. The preparation manifest
records excluded paths and reasons. Both output and frozen destinations must
be empty on the initial run; reruns against populated destinations are refused.

`scripts/verify_fire_split.py` checks test image and label hashes in both copies,
checks split isolation, and writes `results/fire_preparation.md`. As with PPE,
`train_only.yaml` excludes test data and `eval_frozen.yaml` uses the working copy
to keep model cache files outside the frozen snapshot. Only exact byte duplicates
are checked; related video frames and recompressed duplicates may remain.

Fire-model training is a later step. Preparation does not start a second GPU
training job alongside the running PPE baseline and does not measure accuracy.

## Zone editor and monitor

Draw the danger-zone polygon using the actual camera view at the demo location:

```powershell
.\.venv\Scripts\python.exe src/zones.py --draw --source 0
```

Left click adds a corner, right click removes the last corner, Enter saves, and
Escape cancels. Saving creates local `src/zones.json` (ignored by Git) and
`results/zone_preview.png`. Invalid, degenerate, or self-intersecting polygons
are rejected. A generic example is included at `config/zones.example.json`.

Run the live monitor with the pretrained SH17 weights while baseline training
is underway, or use `models/ppe_best.pt` once training finishes:

```powershell
.\.venv\Scripts\python.exe src/zones.py --source 0 --weights models/sh17_yolov8n.pt
```

The monitor tracks each person's bottom-centre point, reports the IDs inside,
continuous intrusion durations, crowd count, and a safe proximity margin once
per second. Press q in the window to stop. Durations use monotonic wall-clock
time; recorded-video playback does not use original media timestamps.

`proximity_margin` is **0 inside/on the zone boundary**, rising linearly to **1
at least 15% of frame width outside**. With no people it is 1. The future risk
engine receives `proximity = 1 - proximity_margin`. This resolves the guide's
conflicting inside-distance descriptions using one consistent safety convention.
Leaving or disappearing from detections resets the person's intrusion timer.

Coordinates are normalized, so resizing the same view preserves the zone.
Redraw it after moving the camera or changing the demo location. The geometry
and timers have automated tests and a two-frame CPU tracking smoke check; live
mouse editing and physical walk-in/out observation remain to be checked by a person.

## Fuzzy Safety Risk Score

`src/risk.py` implements all **27 Stage 1 rules** (PPE + zone + proximity to
hazard) and **9 Stage 2 rules** (hazard + fire evidence to risk). Rules use
min AND, max aggregation, triangular membership functions, and centroid
defuzzification through `skfuzzy.control`. The readable tables and current
verification results are in [docs/risk_rules.md](docs/risk_rules.md).

```powershell
.\.venv\Scripts\python.exe src/risk.py
.\.venv\Scripts\python.exe scripts/verify_risk.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Call `compute_risk(ppe_load, zone_pressure, proximity, fire_evidence,
unhelmeted_in_zone_seconds)` to receive `(score, band)`. The first four inputs
must already be finite numbers in [0,1]; duration must be finite and nonnegative.
Use `proximity = 1 - ZoneMonitor.proximity_margin`. Detection-to-PPE association,
PPE/zone input normalization, and fire-history smoothing are integration work;
this module does not invent those missing definitions or run either detector.

Hard overrides:

- Fire evidence >= 0.7: score 100, Critical; skips fuzzy computation.
- Unhelmeted zone intrusion >= 8 seconds: score at least 85, Critical.
- Unhelmeted zone intrusion >= 3 seconds: score at least 65, High.

Continuous score bands are [0,25] Low, (25,50] Medium, (50,75] High, (75,100]
Critical. This explicitly assigns fractional scores between the guide's integer
boundaries. An override floor never reduces a larger fuzzy result.

**Monotonic scoring:** the raw centroid surfaces have small decreasing regions
despite the ordered rule tables. At initialization the module computes both raw
Mamdani surfaces on a 21-knot grid per input axis, applies prefix maxima along
each axis, and retains immutable corrected tables. Runtime uses multilinear
interpolation. Ordered grid vertices ensure nondecreasing interpolation in
each input; composing the two stages and override floors preserves that property.
The deployed output is therefore a calibrated interpolation of Mamdani centroid
results, **not an unmodified raw centroid at every continuous input**. The audit
records the raw reversals and correction sizes. No detector test images or
expert scenario labels are used to construct these tables.

The risk-engine milestone passed an 18-test suite. Additional engine verification covers 27,300 grid
edges and 10,000 continuous increase comparisons. These are software checks,
not safety-officer agreement or a claim about real-world detection accuracy.
Independent expert-labelled scenarios remain pending. The first call builds
the surfaces; subsequent calls reuse them. Reference:
[scikit-fuzzy control-system example](https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_control_system_advanced.html).

## Phase 7 dashboard and alerts

Start the offline simulation (no camera or detector models required):

```powershell
.\.venv\Scripts\python.exe src/app.py --demo --silent
```

Open **http://127.0.0.1:5050/**. The preview is labelled Demo simulation and
cycles through safe, PPE/zone, fire-evidence, latch-clearing, and crowd scenes.
Its events and snapshots live under `results/demo/`, separate from real logs.
Simulation FPS and latency describe the simulation, not model performance.
Remove `--silent` to enable the laptop alert sound. The current preview was
launched with `--silent` for development review.

The dashboard provides an MJPEG feed, score/band, input bars, processing metrics,
people/zone counts, and ten recent events with snapshot links. It adapts to
desktop and mobile, includes keyboard-focusable table scrolling, and works
without external fonts or network assets. Pause display stops browser updates
only; detection, alerts, and logging continue. Hidden pages suspend browser
polling/streaming. Missing, stale, failed, or hung requests show an unavailable
score rather than a healthy Low state.

### Approved normalization

The user explicitly approved these definitions for the live pipeline:

- `ppe_load = min(1, (2 * missing_helmet_people + missing_vest_people) / (3 * people))`;
  no people gives 0.
- `zone_pressure = 0.5 * min(crowd_count / 2, 1) + 0.5 * min(longest_intrusion_seconds / 5, 1)`;
  no intruders gives 0.
- `proximity = 1 - safe_margin` from the zone monitor.
- `fire_evidence` is the maximum confidence from the latest ten fire checks.
  Fire checks run on the first frame and every ten frames afterward.

Each equipment detection is assigned to at most one compatible person's body
region. Helmet/head centres must be in the upper body; vest centres must be in
the torso. At least half the equipment box must overlap the person. This is a
geometric heuristic, not a validated association model; occlusion, carried PPE,
and overlapping people can still produce mistakes. Missing helmet evidence
contributes to PPE load, but the bare-head duration override requires a detected
head with no associated helmet while inside the zone. It is not inferred solely
from a missing helmet detection. Missing tracking observations reset observed
intrusion timers and cannot count toward confirming or clearing an alert.

### Alert behavior

Risk appears immediately, with Confirming alert while the score is at least
High for less than one second. The buzzer and event qualify after **one continuous
second at High or Critical**; moving between those levels retains the debounce
timer. A Critical alert latches once confirmed. It clears only after **three
continuous seconds below High**. High interruptions or unknown observations reset
that clearing timer. One event is logged per alarm episode, plus one for a later
Critical escalation. Repeated identical frames do not spam events or outputs.

Each event writes a JPEG and a CSV row with timestamp, band, score, four inputs,
and cause. Real output is `results/events.csv` and `results/snapshots/`. UTC
timestamps and unique filenames prevent collisions. Recent history survives
restart; snapshots are served only from the configured snapshot directory.

`AlertDriver(hardware=True)` uses Jetson.GPIO BOARD pin 12 for the buzzer and
pin 16 for the LED. It changes outputs only on level changes and cleans up on
shutdown. This path has mock tests only; physical wiring remains untested.

## Phase 8 measurements

`src/evaluate.py` provides the guide's three measurement modes. Run it only
after both model trainings and frozen-test evaluations finish:

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --map
.\.venv\Scripts\python.exe src/evaluate.py --speed --video path\to\recorded-test.mp4 --zones path\to\site-zones.json --machine "RTX 4060 laptop"
.\.venv\Scripts\python.exe src/evaluate.py --scenarios --scenario-file scenarios\scenarios.csv
```

The speed run requires at least 60 seconds of recorded video and a polygon for
that camera view. It measures the full application pipeline and excludes model
load time; it does not measure physical camera or Jetson performance. The
scenario CSV header is in `scenarios/template.csv`. Add 30 real situations with
expert-selected bands before running that mode. Do not fill it with invented
labels. The scenario mode reports agreement only for rows with expert bands;
fuzzy-policy latency excludes neural inference. `--all` runs the three modes in
order when all required inputs are available. Measured results are written to
`results/final_numbers.md`; missing measurements remain explicitly unreported.

### Live pipeline (models still required)

Once both trained models and the real camera-zone polygon exist:

```powershell
.\.venv\Scripts\python.exe src/app.py --source 0 --device cpu
```

Live mode refuses missing model/zone files. It uses tracked PPE detections each
frame, time-sliced fire checks, zone geometry, approved signals, fuzzy scoring,
debounce/latching, logging, and snapshots. CPU is the default to avoid competing
with ongoing GPU training; use `--device 0` when the GPU is available. The live
camera pipeline has contract tests using model/capture doubles. Full real-model
camera verification remains pending trained fire/PPE weights and zone setup.
Decision latency measures frame acquisition through the risk computation;
processing FPS reflects successive processed frames. Phase 8 benchmarking is
still required before reporting end-to-end performance.

Routes: `/`, `/api/state`, `/video_feed`, `/snapshots/<filename>`.
All **31 tests pass**, including debounce/latch boundaries, snapshot/CSV
consistency, output change suppression, signal normalization, fire scheduling,
route isolation, and failure states. Browser review covered 1280px/375px layouts,
pause behavior, and keyboard table scrolling. See [dashboard design](docs/dashboard-design.md).

## Credits

- [Tanishjain-glitch/PPE-Detection-Jetson](https://github.com/Tanishjain-glitch/PPE-Detection-Jetson): base PPE project, GPL-3.0. Retain upstream notices and license when reusing its code.
- [SH17 by ahmadmughees](https://github.com/ahmadmughees/SH17dataset): dataset and pretrained weights, CC BY-NC-SA 4.0.
- [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset): fire/smoke data. Consult the upstream LICENSE before dataset use or redistribution.
- [Qengineering classic Nano runtime](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano): planned classic-Nano runtime reference.

Upstream repositories remain separate checkouts; their licenses and dataset
terms apply independently. This project is intended to remain open-source,
as required by the guide.
