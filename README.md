# Grounded-SAM Bubble Detection Pipeline

Detect and segment bubbles in X-ray videos using **Grounding DINO** (local, via HuggingFace Transformers) and **SAM 2.1** (Segment Anything Model 2), with **temporal track filtering** to remove transient false positives.

---

## Pipeline Overview

The pipeline uses a two-pass architecture with temporal filtering between passes:

```
Input video
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 1: Load models                    │
│  • Grounding DINO (HuggingFace)         │
│  • SAM 2.1 image predictor              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 2: Extract & crop video frames    │
│  • Decode video → JPEG frames           │
│  • Crop to ROI (bottom 310 rows)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 3–4: Find start frame             │
│  • Coarse scan (10 evenly-spaced        │
│    probes) to find first detection       │
│  • Binary search to refine exact frame  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 5: Pass 1 — Detection only        │
│  • Run Grounding DINO on every frame    │
│  • Store raw bounding boxes + scores    │
│  • No segmentation yet (fast)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 6: Temporal track filtering       │
│  • Link detections across frames        │
│    using IoU-based greedy matching      │
│  • Build tracks (consistent bubble IDs) │
│  • Discard tracks shorter than          │
│    MIN_TRACK_LENGTH frames              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 7: Pass 2 — Segmentation          │
│  • Run SAM 2.1 only on filtered boxes   │
│  • Generate instance masks per frame    │
│  • Save annotated frames + masks        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Steps 8–9: Output                      │
│  • Save detections.json (with track     │
│    IDs, per-bubble info, summary)       │
│  • Compile annotated frames → MP4 video │
└─────────────────────────────────────────┘
```

### Why two passes?

Per-frame detection is noisy — Grounding DINO sometimes produces false positives that appear for 1–2 frames then vanish. By detecting all frames first (Pass 1), building temporal tracks, and filtering out short-lived tracks, we ensure only persistent, real bubbles reach the segmentation step (Pass 2). This also saves GPU time by not running SAM 2 on false positives.

### How tracking works

1. For each frame, compute IoU between every new detection and the last known box of each active track.
2. Greedily match pairs by highest IoU (must exceed `TRACK_IOU_THRESHOLD`).
3. Matched detections extend their track; unmatched detections start new tracks.
4. Tracks not seen for `MAX_TRACK_GAP` consecutive frames are finalized.
5. After all frames, tracks with fewer than `MIN_TRACK_LENGTH` entries are discarded as transient noise.

Each surviving bubble gets a consistent `track_id` across all frames it appears in.

---

## Project Structure

```
Grounded-SAM-pipeline/
├── bubbles_detection_pipeline.py   # Detection pipeline (two-pass + temporal filtering)
├── labeling_pipeline.py            # Frame labeling pipeline (5-category classification)
├── labeling_rules.yaml             # Configurable rules for the labeling pipeline
├── evaluate.py                     # Evaluation against labelme ground truth
├── tune_params.py                  # Automatic hyperparameter grid search
├── setup.py                        # SAM 2 package setup
├── pyproject.toml                  # Build system config
├── checkpoints/
│   ├── download_ckpts.sh           # Download SAM 2.1 model weights
│   └── sam2.1_hiera_*.pt           # SAM 2.1 checkpoints (not tracked)
├── sam2/                           # SAM 2 library source
├── utils/
│   ├── video_utils.py              # Video creation utility
│   ├── detection_utils.py          # Shared detection & tracking functions
│   └── keyhole_detector.py         # Keyhole trajectory utilities (temporal filter, interpolation, smoothing)
└── data/                           # Data directory (not tracked)
    ├── raw/                        # Input videos
    ├── frames/custom_video_frames/ # Extracted & cropped frames
    ├── output/                     # Detection pipeline output
    │   ├── detections.json
    │   ├── masks/
    │   ├── tracking_results/
    │   └── bubbles_groundedSAM.mp4
    ├── labeling/                   # Labeling pipeline output
    │   ├── labeling_results.json   # Per-frame labels, intervals, track metadata
    │   └── frames/                 # Annotated frames with label overlays
    └── labelme/                    # Ground truth annotations
```

---

## Prerequisites

- **NVIDIA GPU** with CUDA support (Ampere or newer recommended)
- Python **3.10+**
- PyTorch **2.1+** with CUDA
- CUDA **11.8+**

---

## Installation

```bash
# 1. Clone
git clone https://github.com/KevinYan-831/Grounded-SAM-pipeline.git
cd Grounded-SAM-pipeline

# 2. Create environment
conda create -n grounded_sam2 python=3.10
conda activate grounded_sam2

# 3. Install PyTorch (match your CUDA version — see pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM 2
pip install -e .

# 5. Install dependencies
pip install transformers supervision opencv-python pillow tqdm numpy scipy

# 6. Download SAM 2.1 checkpoints
cd checkpoints && bash download_ckpts.sh && cd ..
```

Grounding DINO weights are **automatically downloaded** from HuggingFace on first run.

---

## Preparing Data

```bash
mkdir -p data/raw
cp /path/to/your/video.mp4 data/raw/x_ray_video.mp4
```

---

## Configuration

All parameters are at the top of `bubbles_detection_pipeline.py`:

### Detection parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO_PATH` | `./data/raw/x_ray_video.mp4` | Input video path |
| `TEXT_PROMPT` | `"bubble."` | Text prompt for Grounding DINO |
| `GDINO_MODEL_ID` | `IDEA-Research/grounding-dino-base` | HuggingFace model ID |
| `BOX_THRESHOLD` | `0.35` | Confidence threshold for detections (0–1) |
| `MAX_BOX_AREA_RATIO` | `0.07` | Discard boxes larger than this fraction of image area |
| `DETECT_INTERVAL` | `1` | Run detection every N frames (1 = every frame) |
| `CROP_TOP` | `490` | Crop frames starting from this row (800 - 310) |

### Temporal filtering parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_TRACK_LENGTH` | `10` | Minimum frames a track must span to be kept |
| `TRACK_IOU_THRESHOLD` | `0.3` | Minimum IoU to link a detection to an existing track |
| `MAX_TRACK_GAP` | `2` | Max consecutive frames a track can go unmatched before ending |

### SAM 2.1 model options

| Model | Checkpoint | Config |
|-------|-----------|--------|
| Tiny | `sam2.1_hiera_tiny.pt` | `sam2.1_hiera_t.yaml` |
| Small | `sam2.1_hiera_small.pt` | `sam2.1_hiera_s.yaml` |
| Base+ | `sam2.1_hiera_base_plus.pt` | `sam2.1_hiera_b+.yaml` |
| **Large** | `sam2.1_hiera_large.pt` | `sam2.1_hiera_l.yaml` |

---

## Running the Pipeline

```bash
conda activate grounded_sam2
python bubbles_detection_pipeline.py
```

### Output files

| File | Description |
|------|-------------|
| `data/output/detections.json` | Per-frame detection results with track IDs, bounding boxes, confidence scores, mask areas, and summary statistics |
| `data/output/masks/*.npy` | Raw binary instance masks per frame (shape: `[N, H, W]`) |
| `data/output/tracking_results/*.jpg` | Annotated frames with boxes, labels (`bubble T<id>`), and color masks |
| `data/output/bubbles_groundedSAM.mp4` | Final annotated video |

### JSON output format

```json
{
  "video_path": "./data/raw/x_ray_video.mp4",
  "text_prompt": "bubble.",
  "model": "IDEA-Research/grounding-dino-base",
  "box_threshold": 0.35,
  "max_box_area_ratio": 0.07,
  "min_track_length": 10,
  "track_iou_threshold": 0.3,
  "image_size": {"width": 640, "height": 310},
  "total_frames_processed": 1050,
  "frames_with_detections": 850,
  "total_bubbles_detected": 5230,
  "raw_detections_before_filter": 6100,
  "total_tracks": 120,
  "valid_tracks": 15,
  "start_frame_index": 950,
  "tracks": [
    {
      "track_id": 3,
      "num_frames": 200,
      "first_frame": 960,
      "last_frame": 1160
    }
  ],
  "frames": [
    {
      "frame_index": 960,
      "frame_file": "00960.jpg",
      "num_bubbles": 3,
      "bubbles": [
        {
          "id": 0,
          "track_id": 3,
          "label": "bubble",
          "confidence": 0.4521,
          "bbox": [120.5, 80.3, 145.2, 105.8],
          "bbox_area": 630.25,
          "mask_area_pixels": 512
        }
      ]
    }
  ]
}
```

---

## Frame Labeling Pipeline

A separate pipeline that classifies each video frame into one of 5 process states based on keyhole presence and bubble behavior.

### Label categories

| ID | Label | Description |
|----|-------|-------------|
| 0 | No Signal | No keyhole detected at this frame |
| 1 | Normal Process | Keyhole present; **no bubbles exist anywhere** in the entire video |
| 2 | Unstable Process without Pore Generation | Keyhole present; no bubble at this frame, but bubbles exist elsewhere in the video |
| 3 | Transient Pore Generation | At least one bubble at this frame **will disappear** later |
| 4 | Permanent Pore Generation | All bubbles at this frame are permanent (none will disappear) |

#### Labeling priority hierarchy

Labels are assigned in the following order of precedence (first matching rule wins):

```
0  No keyhole detected
↓
1  Keyhole present, zero bubbles in entire video
↓
2  Keyhole present, no bubbles at this frame (but exist elsewhere)
↓
3  Any bubble at this frame is transient  ← transient beats permanent
↓
4  All bubbles at this frame are permanent
```

> **Transient takes priority over permanent (label 3 > label 4).**
> If a frame contains both a transient bubble and a permanent bubble,
> it is labeled 3 — because a new transient pore generation event is occurring.

### How it works

1. **GDINO detection** — Runs Grounding DINO once per frame with the `"bubble.pore"` prompt using a broad threshold to capture both bubbles and the keyhole in a single pass.
2. **Keyhole split (leftmost heuristic)** — Among all detections in a frame, the leftmost box with height ≥ `min_height` is the keyhole candidate. The keyhole sits at the leading edge of the weld pool and moves right→left over time, so it is always the leftmost active feature.
3. **Optional SAM2 refinement** — If `use_sam_refinement: true`, SAM2's point-prompt segmentation is run on each keyhole candidate using its center coordinate, producing a more precise bounding box.
4. **Keyhole trajectory** — Temporal outlier rejection (local median filter on cx) → find first real keyhole frame → linear interpolation of gaps → rolling-average smoothing.
5. **Bubble tracks** — IoU-based tracking builds per-bubble tracks across all frames.
6. **Track classification** — Each bubble track is tagged as transient (`duration < transient_max_frames`) or permanent (`duration ≥ permanent_min_frames`), and near/far relative to the keyhole center.
7. **Per-frame labeling** — The priority hierarchy above assigns one of 5 labels to each frame.
8. **Smoothing & output** — Majority-vote smoothing reduces flickering; results are grouped into labeled time intervals and saved.

### Running

```bash
# If frames are already extracted (from the detection pipeline):
python labeling_pipeline.py --skip-extraction

# Skip GDINO detection too (reuse cached raw_detections.json):
python labeling_pipeline.py --skip-extraction --skip-detection

# Full run (extracts frames from video first):
python labeling_pipeline.py

# Custom config:
python labeling_pipeline.py --config my_rules.yaml --skip-extraction
```

### Configuration (`labeling_rules.yaml`)

| Section | Key parameters |
|---------|---------------|
| `detection.bubble` | `text_prompt`, `box_threshold`, `max_box_area_ratio` |
| `detection.keyhole` | `min_height`, `box_threshold`, `max_box_area_ratio`, `use_sam_refinement` |
| `keyhole_trajectory` | `max_jump_px`, `median_window`, `smoothing_window`, `first_frame_min_x` |
| `tracking` | `track_iou_threshold`, `max_track_gap`, `bubble_min_track_length` |
| `track_classification` | `transient_max_frames`, `permanent_min_frames` |
| `proximity` | `method`, `near_threshold_pixels` |
| `labels` | Label names, IDs, colors, descriptions |
| `intervals` | `smoothing_window`, `min_interval_length` |

**To enable SAM2 keyhole refinement**, set in `labeling_rules.yaml`:
```yaml
detection:
  keyhole:
    use_sam_refinement: true
```
This passes the approximate keyhole center to SAM2 as a point prompt and replaces the GDINO bounding box with the SAM-derived one. More accurate but adds ~5 minutes to the run.

### Output

| File | Description |
|------|-------------|
| `data/labeling/labeling_results.json` | Per-frame labels, time intervals, bubble track metadata, keyhole positions |
| `data/labeling/raw_detections.json` | Cached raw GDINO detections (reused with `--skip-detection`) |
| `data/labeling/frames/*.jpg` | Annotated frames: colored label bar at top, white keyhole box, bubble boxes (red = permanent, orange = transient near keyhole, cyan = far) |

---

## Evaluation & Tuning

### 1. Annotate ground truth

Select random frames from pipeline output and annotate with labelme:

```bash
python evaluate.py --select-frames --num-frames 10
labelme data/labelme/
```

Label bubbles as polygons with the label `bubble`.

### 2. Evaluate current parameters

```bash
python evaluate.py --iou-threshold 0.5
```

Reports pixel-level (precision, recall, F1, IoU) and instance-level (Hungarian matching) metrics.

### 3. Automatic hyperparameter tuning

```bash
python tune_params.py
```

This runs a grid search over 864 parameter combinations in a **single run**:

| Parameter | Values tested |
|-----------|--------------|
| `BOX_THRESHOLD` | 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 |
| `MAX_BOX_AREA_RATIO` | 0.03, 0.05, 0.07, 0.1 |
| `MIN_TRACK_LENGTH` | 1, 3, 5, 10 |
| `TRACK_IOU_THRESHOLD` | 0.2, 0.3, 0.4 |
| `MAX_TRACK_GAP` | 1, 2, 3 |

**How it works efficiently:**
1. Grounding DINO runs **once** on all frames at the lowest threshold, caching every possible detection
2. For each combo, filtering + tracking is replayed in pure Python (instant)
3. SAM2 runs only on the ~20 labelme frames per combo

**Output:** The script reports the best parameter combo for **every metric** in one run — no need to re-run with different flags. Example output:

```
BEST PARAMETERS PER METRIC (single run — no need to re-run)
========================================================

  pixel_precision = 0.9120
    BoxTh=0.40  MaxArea=0.03  MinTrk=10  TrkIoU=0.40  MaxGap=1

  pixel_f1 = 0.8450
    BoxTh=0.25  MaxArea=0.07  MinTrk=5   TrkIoU=0.30  MaxGap=2

  inst_f1 = 0.8100
    BoxTh=0.30  MaxArea=0.05  MinTrk=3   TrkIoU=0.30  MaxGap=2
```

The `--metric` flag only controls which metric the **top-K table** is sorted by (default: `pixel_f1`). All metrics are always computed and shown.

Full results are saved to `data/output/tuning_results.json`.

Copy the best parameters back into `bubbles_detection_pipeline.py` and re-run the pipeline.

---

## Manual Tuning Guide

| Problem | Solution |
|---------|----------|
| Too many false positives | Increase `BOX_THRESHOLD` or `MIN_TRACK_LENGTH` |
| Missing real bubbles | Decrease `BOX_THRESHOLD` or `MIN_TRACK_LENGTH` |
| Large spurious boxes | Decrease `MAX_BOX_AREA_RATIO` |
| Tracks breaking for fast-moving bubbles | Decrease `TRACK_IOU_THRESHOLD` (e.g. 0.15) |
| Tracks breaking due to missed detections | Increase `MAX_TRACK_GAP` (e.g. 5) |
| Out of memory | Use a smaller SAM 2.1 model (tiny/small) |
| Slow processing | Increase `DETECT_INTERVAL` to skip frames |
