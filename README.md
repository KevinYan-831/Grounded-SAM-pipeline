# X-Ray Welding Video Analysis Pipeline

Detect bubbles (pores) and keyholes in X-ray welding videos using **Grounding DINO** + **SAM 2.1**, then classify each frame by the type of pore generation occurring.

---

## How It Works (High Level)

1. User annotates two frames per trajectory in **LabelMe** (keyhole start/end points)
2. Pipeline interpolates keyhole position and segments it with **SAM2** per frame
3. **Grounding DINO** detects bubbles on every frame (separate from keyhole)
4. Bubbles are tracked across frames via IoU matching, with fragmented tracks merged
5. Each frame is labeled based on whether a new bubble is **generated** at that moment

---

## Label Definitions (4 States)

Labels are **generation-based**: they describe whether a new bubble is emerging from the keyhole, not what bubbles are currently visible.

| ID | Label | Description | Color |
|----|-------|-------------|-------|
| 0 | Normal Process | No bubble is ever generated in the entire trajectory | Green |
| 1 | Unstable Process without Pore Generation | Bubbles exist in trajectory, but no generation event at this frame | Yellow |
| 2 | Permanent Pore Generation | A new bubble appearing next frame will persist until the end | Red |
| 3 | Transient Pore Generation | A new bubble appearing next frame will disappear before the end | Orange |

### How generation labeling works

Bubbles emerge from inside the keyhole and can't be detected as separate objects until they move away. So the moment a bubble is first detected by GDINO (frame `t`) is **after** it was actually generated.

For each new bubble first detected at frame `t`, only **frame `t - 1`** receives the generation label. All other frames get label 1 (Unstable).

A birth only counts as a generation event if the bubble's first detection is **near the keyhole** (within `birth_proximity_px`). Bubbles first appearing far from the keyhole are assumed to be re-detections or noise, not newly generated pores.

```
Per-frame logic:
  1. If NO bubbles exist anywhere in the trajectory -> label 0 (Normal)
  2. If this frame is t-1 of a permanent bubble birth (near keyhole) -> label 2
  3. If this frame is t-1 of a transient bubble birth (near keyhole) -> label 3
  4. Otherwise -> label 1 (Unstable)
```

### Permanent vs transient classification

- **Permanent**: bubble track's last detected frame >= `end_frame` (keyhole end annotation)
- **Transient**: bubble track disappears before `end_frame`
- No duration thresholds — classification is purely based on whether the bubble persists to the end of the analyzed range

---

## Handling Detection Inconsistency

GDINO may miss a bubble for a few frames then re-detect it. Without mitigation, re-detections would create false birth events. Three mechanisms prevent this:

1. **IoU tracker gap tolerance** (`max_track_gap: 10`): a track stays active for up to 10 frames without detection, re-matching when the bubble reappears
2. **Track merging** (`max_gap_frames: 30`, `max_distance_px: 50`): rejoins fragmented tracks of the same bubble across longer gaps. Only merges sequential tracks — coexisting bubbles (overlapping in time) are never merged. Also handles **boundary deduplication**: when GDINO produces duplicate detections of the same bubble at a single frame, causing two tracks to "meet" at that frame, the merge step detects spatially close boxes at the boundary and joins the tracks
3. **Consecutive birth filter** (`min_consecutive_birth: 2`): a bubble's birth is the first frame with N+ consecutive detections. Tracks that never achieve N consecutive detections are treated as noise and produce no generation event at all
4. **Birth proximity filter** (`birth_proximity_px: 150`): only counts a birth as a generation event if the bubble first appears near the keyhole. Far-away re-detections that slip past tracking are filtered out

---

## Project Structure

```
Grounded-SAM-pipeline/
├── labeling_pipeline_manual_keyhole.py  # PRIMARY: manual keyhole + auto bubble detection
├── labeling_pipeline.py                 # Auto-keyhole pipeline + shared functions
├── labeling_rules.yaml                  # All configurable parameters
├── run_labeling_batch.py                # Batch runner for multiple trajectories
├── build_transition_model.py            # Aggregate results into transition matrix
├── prepare_keyhole_labelme_workspace.py # Convert TIFF trajectories to LabelMe-ready PNGs
├── bubbles_detection_pipeline.py        # Standalone bubble detection (no labeling)
├── evaluate.py                          # Evaluation against ground truth
├── tune_params.py                       # Hyperparameter grid search
├── setup.py                             # SAM2 package setup
├── utils/
│   ├── detection_utils.py               # Shared: load_models, detect_on_frame, build_and_filter_tracks
│   ├── keyhole_detector.py              # Keyhole trajectory: temporal filter, interpolation, smoothing
│   └── video_utils.py                   # Video creation utility
├── checkpoints/                         # SAM 2.1 model weights (not tracked)
└── sam2/                                # SAM 2 library source
```

---

## Pipelines

### 1. Manual Keyhole Pipeline (`labeling_pipeline_manual_keyhole.py`) — PRIMARY

Used when keyhole auto-tracking is unreliable (which is typical). User provides keyhole start/end points via LabelMe annotations.

**Steps:**
1. Read LabelMe annotations (`keyhole_start`, `keyhole_end` point labels)
2. Linearly interpolate keyhole prompt point between start and end frames
3. Run SAM2 point-prompt segmentation per frame to get keyhole bounding box
   - Tries all 3 SAM masks (highest score first), picks first passing shape validation (narrow + tall)
   - Falls back to previous valid bbox, then to small point-box if all masks fail
4. Run Grounding DINO bubble detection on analyzed frames only
5. Filter bubbles: remove those overlapping keyhole bbox AND those to the LEFT of keyhole center
6. Build IoU-based bubble tracks, merge fragmented tracks (sequential only, never coexisting)
7. Classify tracks as permanent/transient based on whether they persist to `end_frame`
8. Label each frame in `[start_frame, end_frame]` — only births near keyhole count as generation events
9. Output labeled frames and `labeling_results.json`

### 2. Auto-Keyhole Pipeline (`labeling_pipeline.py`)

Fully automatic — detects keyhole by shape heuristic (leftmost tall/narrow GDINO detection). Uses the same labeling logic as the manual pipeline.

### 3. Standalone Detection Pipeline (`bubbles_detection_pipeline.py`)

Raw two-pass bubble detection without labeling. Useful for inspecting what GDINO sees.

---

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA (Ampere+ recommended)
- Python 3.10+, PyTorch 2.1+ with CUDA

### Installation

```bash
# Create environment
conda create -n grounded_sam2 python=3.10
conda activate grounded_sam2

# Install PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SAM 2
pip install -e .

# Install dependencies
pip install transformers supervision opencv-python pillow tqdm numpy scipy pyyaml

# Download SAM 2.1 checkpoints
cd checkpoints && bash download_ckpts.sh && cd ..
```

### Step 1: Prepare frames from TIFF trajectories

```bash
python prepare_keyhole_labelme_workspace.py \
  --data-root /path/to/xray-enhanced \
  --output-root /path/to/xray-enhanced-frames
```

Creates one directory per trajectory with frames `00000.png ... NNNNN.png`.

### Step 2: Annotate keyhole in LabelMe

```bash
labelme /path/to/xray-enhanced-frames/<prefix>/<trajectory_name>
```

On two frames:
- First keyhole frame: one point labeled **`keyhole_start`**
- Last keyhole frame: one point labeled **`keyhole_end`**

(Fallback: label both as `keyhole`; earliest = start, latest = end.)

### Step 3: Run labeling

**Single trajectory:**

```bash
conda run -n grounded_sam2 python labeling_pipeline_manual_keyhole.py \
  --config labeling_rules.yaml \
  --frames-dir /path/to/frames/<trajectory> \
  --output /path/to/output/labeling_results.json \
  --output-frames-dir /path/to/output/frames \
  --skip-extraction
```

**All trajectories (batch):**

```bash
conda run -n grounded_sam2 python run_labeling_batch.py \
  --data-root /path/to/xray-enhanced-frames \
  --output-root /path/to/labeling_results \
  --exts png \
  --manual-keyhole-from-labelme \
  --skip-existing
```

Add `--dry-run` to preview what would run without executing. Unannotated trajectories are automatically skipped.

### Step 4: Build transition model

```bash
python build_transition_model.py \
  --results-root /path/to/labeling_results
```

Aggregates all `labeling_results.json` files into a 4x4 transition probability matrix and per-trajectory CSV summary.

---

## Configuration (`labeling_rules.yaml`)

### Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection.bubble.text_prompt` | `"bubble.pore"` | GDINO text prompt |
| `detection.bubble.box_threshold` | `0.25` | Confidence threshold for bubble detections |
| `detection.bubble.max_box_area_ratio` | `0.03` | Max box area as fraction of image |
| `detection.keyhole.min_height` | `50` | Min height (px) for keyhole shape validation |
| `detection.keyhole.max_width` | `100` | Max width (px) — keyhole must be narrow |
| `detection.keyhole.min_aspect_ratio` | `1.5` | Min h/w ratio — keyhole must be tall |
| `detection.keyhole.use_sam_refinement` | `true` | Refine keyhole bbox with SAM2 point prompt |

### Tracking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracking.track_iou_threshold` | `0.2` | Min IoU to link detections across frames |
| `tracking.max_track_gap` | `10` | Max frames a track can go undetected |
| `tracking.bubble_min_track_length` | `1` | Min frames for a track to be kept |
| `tracking.min_consecutive_birth` | `2` | Min consecutive detections to establish a birth (filters isolated noise) |
| `track_merging.enabled` | `true` | Enable/disable fragmented track merging |
| `track_merging.max_gap_frames` | `30` | Max temporal gap to merge fragmented tracks |
| `track_merging.max_distance_px` | `50` | Max spatial distance to merge tracks |

### Proximity & Birth Filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `proximity.method` | `"center_distance"` | Distance calculation method |
| `proximity.near_threshold_pixels` | `150` | Distance threshold for "near keyhole" classification |
| `proximity.birth_proximity_px` | `150` | Max distance from keyhole for a birth to count as generation |

### Manual Keyhole

| Parameter | Default | Description |
|-----------|---------|-------------|
| `manual_keyhole.fallback_box_half_size_px` | `10` | Half-size of fallback bbox when SAM fails |
| `manual_keyhole.smoothing_window` | `3` | Rolling average on keyhole center trajectory |
| `manual_keyhole.keep_previous_bbox_on_invalid_mask` | `true` | Reuse last valid bbox before point-box fallback |

---

## Output Format

### `labeling_results.json`

```json
{
  "keyhole_detection_method": "manual_labelme_sam",
  "analysis_frame_range": [120, 443],
  "label_definitions": { ... },
  "summary": {
    "Normal Process": 0,
    "Unstable Process without Pore Generation": 283,
    "Permanent Pore Generation": 25,
    "Transient Pore Generation": 16
  },
  "intervals": [
    {"start_frame": 120, "end_frame": 141, "label_id": 2, ...}
  ],
  "tracks": {
    "bubble_tracks": [
      {"track_id": 0, "first_frame": 141, "last_frame": 443, "is_permanent": true, ...}
    ]
  },
  "label_sequence": [1, 1, 2, 1, ...],
  "transition_counts": [[...], [...], [...], [...]],
  "frame_labels": [
    {"frame_index": 120, "label_id": 1, ...}
  ]
}
```

### Annotated frames

Saved to the output frames directory. Each frame has:
- Colored label bar at top (green/yellow/red/orange)
- White keyhole bounding box
- Colored bubble bounding boxes with track IDs (red=permanent, orange=transient)

---

## Key Design Decisions

1. **Manual keyhole annotation** is preferred because auto-detection (GDINO shape heuristic) is unreliable — the keyhole appearance varies across trajectories.

2. **Only bubbles to the RIGHT of the keyhole** are counted. Bubbles emerge from the keyhole and move rightward; detections to the left are noise.

3. **Generation-based labeling** instead of presence-based: the label describes what is being generated from the keyhole, not what is currently visible. This is because at the exact moment of generation, the bubble is still inside the keyhole and undetectable.

4. **No label smoothing**: generation events are sparse (one frame per bubble birth), and majority-vote smoothing would erase them.

5. **Permanent wins over transient** when both types of bubbles are born at the same time, because permanent pore generation is a more significant process event.

6. **Track merging** handles GDINO detection gaps: if a bubble is missed for up to 30 frames but reappears nearby, the tracks are merged. Only sequential (non-overlapping) tracks are merged — coexisting bubbles always keep separate IDs.

7. **Birth proximity filter**: only bubbles first detected near the keyhole count as generation events. This prevents far-away re-detections from creating false birth labels.

---

## Tuning Guide

| Problem | Solution |
|---------|----------|
| Too many false positive bubbles | Increase `detection.bubble.box_threshold` |
| Missing real bubbles | Decrease `detection.bubble.box_threshold` |
| Keyhole SAM mask too wide (includes bubbles) | Decrease `detection.keyhole.max_width` |
| Tracks breaking for fast-moving bubbles | Decrease `tracking.track_iou_threshold` |
| Permanent bubble split into transient fragments | Increase `track_merging.max_gap_frames` |
| Too many short-lived noise tracks | Increase `tracking.bubble_min_track_length` |
| Re-detections causing false birth labels | Increase `tracking.max_track_gap` or `track_merging.max_gap_frames` |
| Single-frame noise triggering false births | Increase `tracking.min_consecutive_birth` |
| Far-away re-detections triggering generation labels | Decrease `proximity.birth_proximity_px` |

For systematic tuning, use `tune_params.py` (grid search over detection/tracking parameters against LabelMe ground truth) and `evaluate.py` (precision/recall/F1 evaluation).
