# Bubbles Detection Pipeline — User Guide

This guide walks you through setting up and running `bubbles_detection_pipeline.py`, which combines **Grounding DINO 1.5** (via DDS Cloud API) and **SAM 2** to detect and track bubbles in X-ray video footage.

---

## Table of Contents

1. [Overview](#overview)
2. [How the Pipeline Works](#how-the-pipeline-works)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Obtaining a DDS Cloud API Token](#obtaining-a-dds-cloud-api-token)
6. [Preparing Your Data](#preparing-your-data)
7. [Configuration](#configuration)
8. [SAM 2 Model Options](#sam-2-model-options)
9. [Running the Pipeline](#running-the-pipeline)
10. [Understanding the Output](#understanding-the-output)
11. [Tuning Parameters](#tuning-parameters)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline performs the following end-to-end workflow:

```
Input video  →  Frame extraction  →  Frame cropping
     ↓
Grounding DINO 1.5 (cloud)  →  Bounding boxes on frame 0
     ↓
SAM 2 image predictor  →  Segmentation masks on frame 0
     ↓
SAM 2 video predictor  →  Propagate masks across all frames
     ↓
Supervision annotator  →  Annotated frames  →  Output video
```

---

## How the Pipeline Works

| Step | What happens |
|------|-------------|
| **1** | Video is decoded and saved as individual JPEG frames. Each frame is then cropped to the bottom 310 rows (the region of interest in the X-ray). |
| **2** | The first frame is sent to **Grounding DINO 1.5 Pro** (DDS Cloud API) with the text prompt `"bubbles."` to detect bounding boxes around all visible bubbles. |
| **3** | The detected boxes are passed to **SAM 2 Image Predictor** to generate precise segmentation masks for each bubble on frame 0. |
| **4** | Those masks (or boxes/points, depending on `PROMPT_TYPE_FOR_VIDEO`) are registered with the **SAM 2 Video Predictor**, which propagates them forward through every frame of the video. |
| **5** | Each frame is annotated with bounding boxes, labels, and colour masks using the `supervision` library. |
| **6** | Annotated frames are compiled back into an output MP4 video. |

---

## Prerequisites

### Hardware
- **NVIDIA GPU** with CUDA support (Ampere architecture, i.e. RTX 30xx / A100 or newer, is recommended for TF32 acceleration)
- At least **8 GB VRAM** (16 GB+ recommended for longer videos)

### Software
- Python **3.10+**
- CUDA **11.8+** (CUDA 12.x recommended)
- PyTorch **2.1+** with CUDA support

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KevinYan-831/Grounded-SAM-pipeline.git
cd Grounded-SAM-pipeline
```

### 2. Create and activate a conda environment

```bash
conda create -n grounded_sam2 python=3.10
conda activate grounded_sam2
```

### 3. Install PyTorch (with CUDA)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command matching your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install SAM 2

```bash
pip install -e .
```

### 5. Install remaining dependencies

```bash
pip install dds-cloudapi-sdk supervision opencv-python pillow tqdm numpy
```

### 6. Download SAM 2 checkpoints

```bash
cd checkpoints
bash download_ckpts.sh
cd ..
```

This downloads the following model weights into `./checkpoints/`:

| File | Size | Speed |
|------|------|-------|
| `sam2.1_hiera_tiny.pt` | ~38 MB | Fastest |
| `sam2.1_hiera_small.pt` | ~46 MB | Fast (**default**) |
| `sam2.1_hiera_base_plus.pt` | ~80 MB | Balanced |
| `sam2.1_hiera_large.pt` | ~224 MB | Most accurate |

---

## Obtaining a DDS Cloud API Token

The pipeline uses **Grounding DINO 1.5 Pro** through the hugging face token. Get the token and replace it within the code

---

## Preparing Your Data

### Directory structure

The pipeline expects and creates the following layout:

```
Grounded-SAM-2/
├── data/
│   ├── raw/
│   │   └── x_ray_video.mp4       ← your input video goes here
│   ├── frames/
│   │   └── custom_video_frames/  ← auto-created: extracted & cropped frames
│   └── output/
│       ├── bubbles_groundedSAM.mp4  ← final annotated video
│       └── tracking_results/        ← per-frame annotated JPEGs
```

Create the required directories before running:

```bash
mkdir -p data/raw data/frames/custom_video_frames data/output/tracking_results
```

### Input video requirements

- Format: any format OpenCV/supervision can decode (MP4, AVI, MOV, etc.)
- Resolution: the script assumes frame height ≥ 800 px (it crops the bottom 310 rows from an 800 px tall frame). Adjust `CROP_TOP` if your video has a different height — see [Configuration](#configuration).
- Frame rate: any; all frames are processed.

Copy or symlink your input video:

```bash
cp /path/to/your/video.mp4 data/raw/x_ray_video.mp4
```

---

## Configuration

All tunable parameters are at the top of `bubbles_detection_pipeline.py`:

```python
VIDEO_PATH             = "./data/raw/x_ray_video.mp4"
TEXT_PROMPT            = "bubbles."
OUTPUT_VIDEO_PATH      = "./data/output/bubbles_groundedSAM.mp4"
SOURCE_VIDEO_FRAME_DIR = "./data/frames/custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./data/output/tracking_results"
API_TOKEN_FOR_GD1_5    = "<YOUR_DDS_API_TOKEN>"
PROMPT_TYPE_FOR_VIDEO  = "mask"   # "point" | "box" | "mask"
BOX_THRESHOLD          = 0.2
IOU_THRESHOLD          = 0.8
```

| Parameter | Description |
|-----------|-------------|
| `VIDEO_PATH` | Path to the input video file |
| `TEXT_PROMPT` | Natural-language description of the objects to detect. Use a period at the end (e.g. `"bubbles."`) |
| `OUTPUT_VIDEO_PATH` | Where the final annotated video is saved |
| `SOURCE_VIDEO_FRAME_DIR` | Where extracted frames are stored temporarily |
| `SAVE_TRACKING_RESULTS_DIR` | Where per-frame annotated images are saved |
| `API_TOKEN_FOR_GD1_5` | Your DDS Cloud API token |
| `PROMPT_TYPE_FOR_VIDEO` | How SAM 2 video predictor is prompted (see below) |
| `BOX_THRESHOLD` | Confidence threshold for Grounding DINO detections (0–1). Lower = more detections, higher = fewer but more confident |
| `IOU_THRESHOLD` | IoU threshold used for NMS in Grounding DINO (0–1). Higher = more overlapping boxes kept |

### Adjusting the crop region

The script crops each frame to keep only the **bottom 310 rows** (starting from row 490 of an 800 px tall frame):

```python
CROP_TOP = 800 - 310  # = 490
```

If your video has a different height (e.g. 1080 px) and you want the bottom 400 rows:

```python
CROP_TOP = 1080 - 400  # = 680
```

If you do **not** want any cropping, set:

```python
CROP_TOP = 0
```

---

## SAM 2 Model Options

The default model is `sam2.1_hiera_small`. To use a different size, change both lines:

```python
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg       = "configs/sam2.1/sam2.1_hiera_s.yaml"
```

| Model | Checkpoint | Config suffix | Notes |
|-------|-----------|---------------|-------|
| Tiny | `sam2.1_hiera_tiny.pt` | `_t` | Fastest, lowest VRAM |
| **Small** | `sam2.1_hiera_small.pt` | `_s` | **Default — good balance** |
| Base+ | `sam2.1_hiera_base_plus.pt` | `_b+` | Higher accuracy |
| Large | `sam2.1_hiera_large.pt` | `_l` | Best accuracy, most VRAM |

Config paths follow the pattern `configs/sam2.1/sam2.1_hiera_<suffix>.yaml`.

---

## Running the Pipeline

Activate your environment, then run:

```bash
conda activate grounded_sam2
cd /path/to/Grounded-SAM-2
python bubbles_detection_pipeline.py
```

Expected console output (in order):

```
VideoInfo(...)
Saving Video Frames: 100%|████| N/N [...]
Cropping Frames: 100%|████| N/N [...]
[[x1 y1 x2 y2] ...]     # detected bounding boxes
['bubbles', 'bubbles', ...]
# SAM 2 propagation (no explicit progress bar)
Video saved at ./data/output/bubbles_groundedSAM.mp4
```

---

## Understanding the Output

### `data/output/bubbles_groundedSAM.mp4`
The final video with each detected bubble annotated with:
- A **bounding box**
- A **label** (`bubbles`)
- A semi-transparent **colour mask** over the bubble region

### `data/output/tracking_results/annotated_frame_NNNNN.jpg`
Individual annotated frames (zero-padded 5-digit index), one per video frame. Useful for inspecting results frame-by-frame.

### `data/frames/custom_video_frames/NNNNN.jpg`
Extracted and cropped source frames used as SAM 2 input.

---

## Tuning Parameters

### Too many false detections
- Increase `BOX_THRESHOLD` (e.g. `0.35` or `0.5`)
- Increase `IOU_THRESHOLD` to suppress overlapping boxes more aggressively

### Missing bubbles
- Decrease `BOX_THRESHOLD` (e.g. `0.1`)
- Make the `TEXT_PROMPT` more descriptive, e.g. `"small round air bubble."`

### Tracking drift over time
- Switch `PROMPT_TYPE_FOR_VIDEO` from `"mask"` to `"box"` or `"point"` to see which gives better propagation for your video
- Use a larger SAM 2 model (base+ or large)

### Out-of-memory (OOM) errors
- Use a smaller SAM 2 model (`tiny` or `small`)
- Reduce video resolution before running
- Process a shorter clip first to validate the setup

### Prompt type comparison

| `PROMPT_TYPE_FOR_VIDEO` | Description | When to use |
|------------------------|-------------|-------------|
| `"mask"` | Uses full segmentation mask from frame 0 | Best for complex shapes — **default** |
| `"box"` | Uses bounding box from frame 0 | Good for rectangular objects |
| `"point"` | Samples 10 positive points from the mask | Useful when mask quality is uncertain |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'sam2'`
Make sure you installed the package in editable mode from the repo root:
```bash
pip install -e .
```

### `ModuleNotFoundError: No module named 'dds_cloudapi_sdk'`
```bash
pip install dds-cloudapi-sdk
```

### `FileNotFoundError` for checkpoint
Ensure the checkpoint file exists at the configured path:
```bash
ls checkpoints/sam2.1_hiera_small.pt
```
If missing, re-run the download script:
```bash
bash checkpoints/download_ckpts.sh
```

### `AssertionError` or empty `input_boxes`
Grounding DINO found no objects. Try:
- Lowering `BOX_THRESHOLD`
- Rephrasing `TEXT_PROMPT`
- Verifying your API token is valid and has remaining quota

### `CUDA out of memory`
Switch to a smaller model or reduce video length/resolution.

### API authentication error
Double-check `API_TOKEN_FOR_GD1_5`. Tokens expire or may have usage limits on the DDS Cloud platform.

### Output video is black or blank
The `mp4v` codec may not be supported by all players. Try opening with VLC or converting:
```bash
ffmpeg -i data/output/bubbles_groundedSAM.mp4 -vcodec libx264 data/output/bubbles_groundedSAM_h264.mp4
```
