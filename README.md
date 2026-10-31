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

The background controller `scripts/run_public_baseline.ps1` is already started.
It installs pinned CUDA PyTorch wheels from the official PyTorch index, checks
GPU computation, saves `requirements-gpu-lock.txt`, then runs `src/train.py` on
the RTX 4060. Settings: 640 pixels, batch 8, up to 60 epochs, patience 15,
two data workers, seed 42, and no RAM image cache. Windows training uses a main
guard. Reference: [Ultralytics training](https://docs.ultralytics.com/modes/train/)
and [PyTorch GPU setup](https://pytorch.org/get-started/locally/).

Progress files (do not start a duplicate controller):

- `results/baseline-setup.json`: GPU installation / training stage.
- `results/public-baseline.log` and `results/public-baseline-errors.log`: controller output.
- `results/ppe-training-status.json`: epoch and validation progress once training starts.
- `results/ppe_public_baseline/weights/`: training checkpoints.
- `models/ppe_best.pt`: copied only after a successful completed run.

If interrupted after a checkpoint is saved, resume from the project directory:

```powershell
.\.venv\Scripts\python.exe src/train.py --resume results/ppe_public_baseline/weights/last.pt
```

Resume only after checking that the previous process has stopped. Final test
evaluation is a separate step; no accuracy result is available during setup.

The post-training evaluator is also running in the background. It waits for
successful completion, verifies frozen test hashes, and calls `src/train_report.py`
once. Progress is in `results/post-training-status.json`; output is logged in
`results/post-training.log` and `results/post-training-errors.log`. A successful
evaluation produces `results/ppe_eval.md`, `results/ppe_eval.json`, and
`results/confusion_matrix.png`, using measured values from Ultralytics' API.
It refuses to overwrite an existing test report. If training fails or is
interrupted, the watcher stops so the run can be inspected before resuming.

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

## Credits

- [Tanishjain-glitch/PPE-Detection-Jetson](https://github.com/Tanishjain-glitch/PPE-Detection-Jetson): base PPE project, GPL-3.0. Retain upstream notices and license when reusing its code.
- [SH17 by ahmadmughees](https://github.com/ahmadmughees/SH17dataset): dataset and pretrained weights, CC BY-NC-SA 4.0.
- [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset): fire/smoke data. Consult the upstream LICENSE before dataset use or redistribution.
- [Qengineering classic Nano runtime](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano): planned classic-Nano runtime reference.

Upstream repositories remain separate checkouts; their licenses and dataset
terms apply independently. This project is intended to remain open-source,
as required by the guide.
