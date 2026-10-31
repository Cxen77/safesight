# Build progress

Started: 2026-10-31 (Asia/Calcutta).

## Phase 0 — Setup

- [x] Read the complete build guide.
- [x] Create all requested folders and seven empty Python module placeholders.
- [x] Install Python 3.11.9 alongside the existing Python 3.15.
- [x] Create `.venv` with Python 3.11.
- [x] Add requirements, README, GPL-3.0 license, and ignore rules.
- [x] Clone `base-ppe`, `base-sh17`, and `data/dfire`.
- [x] Download SH17 release `v1/yolo8n.pt` as `models/sh17_yolov8n.pt`.
- [x] Copy `base-ppe/Weights/best.pt` as `models/ppe_base_best.pt`.
- [x] Read D-Fire's LICENSE before dataset use.
- [x] Install and import all seven required Python packages; `pip check` passes.
- [x] Verify both weights load successfully through Ultralytics YOLO.

**Phase 0 complete.** GPU setup and baseline training are tracked below.

Evidence: `results/setup-install.log`, `results/upstream-sources.json`, and
`results/model-hashes.json`, and `results/setup-verification.json`. Installed
versions are captured in `requirements-lock.txt`. No model metrics have been
measured yet.

Model inspection confirms SH17 includes `person`, `head`, `helmet`, and
`safety-vest`. The base PPE weights have six classes and do not include person
or head, so use SH17 weights for the planned Phase 1 person-counting demo.
Use the loaded model's class map rather than the ordering in the upstream README.

## Phase 1 — Inference demo

- [x] Implement `src/infer.py` under 120 lines, including colors, class list,
  person count, and a rolling 30-frame processing FPS overlay.
- [x] Support webcam, video, single image, and image folders.
- [x] Save annotated video (including webcam sessions); release resources on exit.
- [x] Process 30 frames of the upstream sample clip with the SH17 weights.
- [x] Decode all 30 saved frames in `results/demo_output.mp4` and inspect preview.
- [x] Process a two-image folder and save both annotated images.
- [x] Verify missing weights and input/output overwrite protection.
- [ ] Observe live webcam detection with a person in view; press q to finish.

Evidence: `results/phase1-video-check.log`, `results/phase1-images-check.log`,
and `results/phase1-preview.jpg`. The upstream clip already contains annotations,
so it is a functional smoke check, not an accuracy benchmark or our own pitch video.
No live webcam FPS claim is made. The overlay reports processing FPS, not the
full end-to-end measurement required in Phase 8.

Run instructions: see README. Webcam observation remains the Phase 1 gate.

## Data acquisition pending (Phase 2 / Phase 4)

Cloning the dataset repositories does **not** download their training images.
Both public archive downloads are complete and CRC-checked during extraction:
SH17 (14,096,291,832 archive bytes, 32,398 entries) in `data/sh17/`, and
D-Fire (3,049,605,157 archive bytes, 43,055 entries) in `data/dfire/dataset/`.
Evidence: `data/downloads/sh17-status.json` and
`data/downloads/dfire-status.json`, both state `complete`.
Raw data is available. Public PPE preparation and the baseline split are complete.
Own lab photos are unavailable and optional under the updated public-only plan.

- SH17 is prepared in `data/ppe_public/`, with `safety-vest` remapped to `vest`.
- D-Fire is downloaded and audited. Its LICENSE remains in the source checkout.
- Own data: unavailable; user approved a public-only baseline, with variable
  demonstration surroundings. Site adaptation can be added later if possible.
- Public PPE test data is frozen in `data/splits_frozen/ppe/` and must not change.

## Phase 2 — Public-only baseline preparation complete

- [x] Implement merge/split/report commands in `src/dataset_tools.py`.
- [x] Validate all SH17 labels and remap classes to person=0, head=1,
  helmet=2, vest=3; keep empty labels for negative images.
- [x] SHA-256 check all 8,099 SH17 images; no exact duplicates found.
- [x] Create `data/ppe_public/` with hardlinked images and independent labels.
  Do not edit these images in place: they share storage with source files.
- [x] Generate a seed-42 multilabel split preview; no final split was written.
- [x] Verify rare helmet/vest image proportions stay near 70/15/15.
- [x] Pass five tests covering class remapping, duplicate protection,
  repeatability, rare-class balance, dry-run behavior, and freeze protection.
- [x] Audit D-Fire's existing splits; record 26 invalid label files (18 files
  with nonpositive box dimensions, eight with invalid normalized coordinates).
  Raw labels and original split assignments are unchanged.
- [x] Adopt public-only data following the user's confirmation that site photos
  are unavailable and demo surroundings may change.
- [x] Create 5,669 train / 1,214 validation / 1,216 test images with seed 42.
- [x] Freeze an independent test copy and save image/label hash manifests.
- [x] Verify frozen and working test copies match and no exact image hash
  appears across the baseline splits.
- [ ] Additional domain-shift checks on separate surroundings after training;
  near-duplicate and recording-session overlap are not yet audited.

Public counts: person 13,802; head 11,985; helmet 927; vest 530 boxes.
All projected training classes exceed 50 boxes. These are dataset counts, not
model performance measurements. The original SH17 checkpoint was pretrained
on SH17, so the new baseline uses general COCO YOLOv8n weights instead.
No site-specific generalization or cross-pretraining-dataset independence is claimed.

Evidence: `results/dataset_preparation.md`, `results/ppe-public-inventory.txt`,
`results/ppe-public-split-preview.json`, and `results/dfire-label-issues.json`.
Rerun the preview with `python scripts/preview_ppe_split.py`.
The live webcam observation in Phase 1 remains unverified; public-data
preparation continued at the user's request without marking that gate complete.

## Phase 3 — GPU verified; baseline training in progress

- [x] Detect NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM, and 16 GB system RAM.
- [x] Discover installed PyTorch is CPU-only; start installing CUDA-enabled
  torch 2.11.0+cu128 / torchvision 0.26.0+cu128 from the official PyTorch index.
- [x] Download and load general COCO YOLOv8n (80 classes) as starting weights.
- [x] Add `src/train.py` and a background setup/training controller.
- [x] Separate training-only and test-evaluation dataset descriptors.
- [x] Finish CUDA package installation and verify GPU matrix multiplication
  with torch 2.11.0+cu128; record `results/gpu-verification.json`.
- [ ] Train: 640 pixels, batch 8, up to 60 epochs, patience 15, workers 2.
- [ ] Save successful best checkpoint as `models/ppe_best.pt`.
- [ ] Evaluate the frozen test split once training is complete.
- [x] Implement `src/train_report.py`: per-class API metrics, mAP@0.50 and
  mAP@0.50:0.95, confusion matrix, and weak-class photo-collection suggestions.
- [x] Pass seven tests, including metric class-index alignment, missing-class
  handling, recommendation boundaries, and rejection of invalid metric values.
- [x] Start `scripts/evaluate_when_ready.py` to evaluate once after successful
  training completion; it stops if training reports failure or interruption.

The first epoch has completed, and the second is running on the RTX 4060.
Training progress files contain validation metrics, not final test accuracy.
The test report is not generated until training completes and test validation runs.

Controller stages are recorded in `results/baseline-setup.json`; epoch progress
will appear in `results/ppe-training-status.json` once training begins. Logs:
`results/public-baseline.log` and `results/public-baseline-errors.log`.
No accuracy numbers have been measured or reported yet. Do not launch a
second controller while the recorded process is active.
The evaluation watcher has its own `results/post-training.pid`,
`results/post-training-status.json`, and `results/post-training.log`.
Successful outputs will be `results/ppe_eval.md`, `results/ppe_eval.json`, and
`results/confusion_matrix.png`. Frozen hashes are verified before evaluation;
YOLO uses the nonfrozen working test copy for its cache files.

## Phase 4 — Fire dataset preparation

Prepared while the PPE run continues; no second GPU training job was launched.

- [x] Add `src/prepare_fire.py` with whole-pair quarantine for invalid labels.
- [x] Preserve source split assignments and the declared IDs: 0 smoke, 1 fire.
- [x] Retain valid empty-label negative images without editing source files.
- [x] Scan image hashes for duplicates within and across all source splits;
  none found in this download.
- [x] Exclude 26 invalid pairs: 13 train, one validation, 12 test.
- [x] Prepare 14,109 train / 3,098 validation / 4,294 test images.
- [x] Create an independent fire test snapshot in `data/splits_frozen/fire/`.
- [x] Write training-only and working-copy evaluation YAML descriptors.
- [x] Pass eight tests, including quarantine, duplicate split isolation,
  retained negatives, source preservation, and frozen-copy independence.
- [x] Verify all 4,294 frozen and working test image/label pairs by SHA-256;
  no exact cross-split image duplicates remain.
- [ ] Train the fire model after the PPE run and evaluation finish.
- [ ] Evaluate fire/smoke performance and record measured values.

Data preparation evidence: `data/dataset_fire/preparation_manifest.json` and
`results/fire-preparation.log`, `results/fire-split-verification.json`, and
`results/fire_preparation.md`. The raw D-Fire audit still records its original
invalid files; those files are excluded from the prepared dataset rather than
modified. Source counts and prepared counts therefore differ intentionally.

## Phase 5 — Zone geometry implemented

- [x] Add normalized polygon editor (left add, right undo, Enter save, Escape cancel).
- [x] Reject invalid, degenerate, and self-intersecting polygons.
- [x] Add tracked-person feet tests, monotonic intrusion timers, and crowd counts.
- [x] Reset timers when a person leaves or disappears from detections.
- [x] Define safe margin: 0 inside/on boundary; linear distance outside, capped
  at 1 beyond 15% frame width; no people means margin 1.
- [x] Test resolution scaling, boundary inclusion, durations, crowd count,
  invalid inputs, and safe-margin behavior. All 12 project tests pass.
- [x] Run a two-frame CPU check with SH17 + ByteTrack: four stable person IDs,
  two intruders, and increasing intrusion timers in the example polygon.
- [x] Install and record the tracker dependency `lap`.
- [ ] Observe the live zone editor and physical walk-in/out demo at the actual site.

The pipeline risk input will be `proximity = 1 - proximity_margin`. Recorded
video timers use processing wall time, not source media timestamps. Redraw the
polygon whenever the camera moves or the demonstration site changes.
Smoke-check evidence: `results/zones-tracking-smoke.json` (functional check,
not a detector accuracy or performance benchmark).

## GitHub

Repository: https://github.com/Cxen77/safesight, branch `main`.
Initial project code was committed and pushed. The user authorized continued
pushes of completed steps. Datasets, weights, model exports, generated results,
virtual environments, and local zone configurations are ignored.

## Phase 6 — Fuzzy risk engine implemented and verified

- [x] Implement `compute_risk` with score and band output and validated inputs.
- [x] Create all 27 Stage 1 rules and all nine Stage 2 rules using
  `skfuzzy.control`, triangular memberships, min/max inference, and centroid.
- [x] Apply fire >= 0.7 -> 100, intrusion >= 8s -> floor 85, >= 3s -> floor 65.
- [x] Verify floors do not lower a stronger fuzzy result; reject NaN, infinity,
  out-of-range normalized inputs, and negative durations.
- [x] Pass all five examples from the build guide.
- [x] Discover and report raw centroid reversals. Add explicitly documented
  monotonic grid correction and interpolation to meet the increasing-risk requirement.
- [x] Check 27,300 grid edges and 10,000 continuous increases with no decreases.
- [x] Pass all 18 project tests, including full rule coverage, band boundaries,
  exact override thresholds, and monotonicity.
- [ ] Independent expert-labelled scenario validation (not fabricated).
- [ ] Integration with detection association, normalized inputs, and fire history.

The deployed score is a calibrated interpolation of two Mamdani centroid
surfaces. It is not raw centroid defuzzification at every continuous input.
This distinction is recorded in the source, README, and readable rule audit.
Raw table corrections, before stage composition: maximum hazard correction
0.04581 on [0,1]; maximum Stage 2 correction 1.6501 on [0,100]. These are
algorithmic knot corrections, not measured detector error or scenario accuracy.

Evidence: `results/risk-self-test.log`, `results/risk-verification.json`,
`results/risk_rules.md`, and `results/test-suite.log`.
The committed readable audit is `docs/risk_rules.md`.

## Design questions to resolve when their phase begins

- Phase 2 annotation text says "THREE boxes" but lists person + helmet and
  explicitly excludes head; confirm the intended two-box policy.
- Phase 5 proximity convention is resolved above; use its inverse for risk.
- Phase 6 needs numerical definitions for weighted/clipped PPE load and
  crowd/duration zone pressure before dashboard integration in Phase 7.
- Continuous monotonicity is addressed by the Phase 6 correction and checks.
- Never fill evaluation results or expert scenario labels with invented values.
