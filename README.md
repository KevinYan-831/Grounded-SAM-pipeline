# Grounded-SAM Bubble Detection Pipeline

Detect and segment bubbles in X-ray videos using **Grounding DINO** (local, via HuggingFace Transformers) and **SAM 2.1** (Segment Anything Model 2).

---

## How It Works

```
Input video  →  Frame extraction  →  Frame cropping (ROI)
     ↓
Per-frame Grounding DINO detection  →  Bounding boxes
     ↓
Per-frame SAM 2.1 segmentation  →  Instance masks
     ↓
JSON results + annotated frames  →  Output video
```

Each frame is independently processed — no tracking, fresh detections on every frame. This ensures newly appearing bubbles are always detected.

---

## Project Structure

```
Grounded-SAM-pipeline/
├── bubbles_detection_pipeline.py   # Main detection + segmentation pipeline
├── evaluate.py                     # Evaluation script (labelme ground truth)
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
    │   ├── detections.json         # Per-frame detection results
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

### 1. Clone and set up environment

```bash
git clone https://github.com/KevinYan-831/Grounded-SAM-pipeline.git
cd Grounded-SAM-pipeline

conda create -n grounded_sam2 python=3.10
conda activate grounded_sam2
```

### 2. Install PyTorch (match your CUDA version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install SAM 2

```bash
pip install -e .
```

### 4. Install dependencies

```bash
pip install transformers supervision opencv-python pillow tqdm numpy scipy
```

### 5. Download SAM 2.1 checkpoints

```bash
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO_PATH` | `./data/raw/x_ray_video.mp4` | Input video |
| `TEXT_PROMPT` | `"bubble."` | Detection prompt for Grounding DINO |
| `GDINO_MODEL_ID` | `IDEA-Research/grounding-dino-base` | HuggingFace model ID |
| `BOX_THRESHOLD` | `0.2` | Confidence threshold (0-1) |
| `MAX_BOX_AREA_RATIO` | `0.05` | Filter boxes larger than 5% of image area |
| `DETECT_INTERVAL` | `1` | Run detection every N frames |
| `CROP_TOP` | `490` | Crop frames from this row (800-310) |

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

### Output

- **`data/output/detections.json`** — Per-frame detection results with bounding boxes, confidence scores, mask areas, and summary statistics
- **`data/output/masks/`** — Raw binary masks as `.npy` files (one per frame)
- **`data/output/tracking_results/`** — Annotated frame images with boxes, labels, and masks
- **`data/output/bubbles_groundedSAM.mp4`** — Final annotated video

### JSON output format

```json
{
  "video_path": "./data/raw/x_ray_video.mp4",
  "text_prompt": "bubble.",
  "model": "IDEA-Research/grounding-dino-base",
  "box_threshold": 0.2,
  "total_frames_processed": 1000,
  "frames_with_detections": 850,
  "total_bubbles_detected": 5230,
  "frames": [
    {
      "frame_index": 950,
      "frame_file": "00950.jpg",
      "num_bubbles": 3,
      "bubbles": [
        {
          "id": 0,
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

The evaluation script compares pipeline output against labelme ground truth annotations.

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

Reports pixel-level (precision, recall, F1, IoU) and instance-level (Hungarian matching) metrics.

---

## Tuning

| Problem | Solution |
|---------|----------|
| Too many false detections | Increase `BOX_THRESHOLD` (e.g. 0.35) |
| Missing bubbles | Decrease `BOX_THRESHOLD` (e.g. 0.1) |
| Large false-positive boxes | Decrease `MAX_BOX_AREA_RATIO` |
| Out of memory | Use a smaller SAM 2 model (tiny/small) |
| Slow processing | Increase `DETECT_INTERVAL` to skip frames |
