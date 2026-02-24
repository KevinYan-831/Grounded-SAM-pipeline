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
├── bubbles_detection_pipeline.py   # Main pipeline (two-pass + temporal filtering)
├── evaluate.py                     # Evaluation against labelme ground truth
├── setup.py                        # SAM 2 package setup
├── pyproject.toml                  # Build system config
├── checkpoints/
│   ├── download_ckpts.sh           # Download SAM 2.1 model weights
│   └── sam2.1_hiera_*.pt           # SAM 2.1 checkpoints (not tracked)
├── sam2/                           # SAM 2 library source
├── utils/
│   └── video_utils.py              # Video creation utility
└── data/                           # Data directory (not tracked)
    ├── raw/                        # Input videos
    ├── frames/custom_video_frames/ # Extracted & cropped frames
    ├── output/
    │   ├── detections.json         # Detection results with track IDs
    │   ├── masks/                  # Raw mask .npy files per frame
    │   ├── tracking_results/       # Annotated frame images
    │   └── bubbles_groundedSAM.mp4 # Output video
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

## Evaluation

Compare pipeline output against labelme ground truth annotations.

### 1. Select random frames for annotation

```bash
python evaluate.py --select-frames --num-frames 10
```

This copies random frames to `data/labelme/`. Then annotate with labelme:

```bash
labelme data/labelme/
```

Label bubbles as polygons with the label `bubble`.

### 2. Run evaluation

```bash
python evaluate.py --iou-threshold 0.5
```

Reports pixel-level (precision, recall, F1, IoU) and instance-level (Hungarian matching) metrics per frame and averaged.

---

## Tuning

| Problem | Solution |
|---------|----------|
| Too many false positives | Increase `BOX_THRESHOLD` or `MIN_TRACK_LENGTH` |
| Missing real bubbles | Decrease `BOX_THRESHOLD` or `MIN_TRACK_LENGTH` |
| Large spurious boxes | Decrease `MAX_BOX_AREA_RATIO` |
| Tracks breaking for fast-moving bubbles | Decrease `TRACK_IOU_THRESHOLD` (e.g. 0.15) |
| Tracks breaking due to missed detections | Increase `MAX_TRACK_GAP` (e.g. 5) |
| Out of memory | Use a smaller SAM 2.1 model (tiny/small) |
| Slow processing | Increase `DETECT_INTERVAL` to skip frames |
