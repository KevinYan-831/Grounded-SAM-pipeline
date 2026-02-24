"""
Hyperparameter tuning for the Grounded-SAM bubble detection pipeline.

Strategy:
  1. Run Grounding DINO ONCE on all frames with a low threshold → cache all detections
  2. For each parameter combo: re-filter + re-track in pure Python (fast)
  3. Run SAM2 only on labelme frames for each combo's filtered boxes
  4. Evaluate against ground truth, report best combos

Usage:
  python tune_params.py
  python tune_params.py --metric pixel_f1
  python tune_params.py --metric inst_f1
"""

import os
import json
import argparse
import itertools
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from evaluate import labelme_polygons_to_masks, evaluate_frame

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- Paths (match the pipeline) ----------
SOURCE_VIDEO_FRAME_DIR = "./data/frames/custom_video_frames"
LABELME_DIR = "./data/labelme"
RESULTS_JSON_PATH = "./data/output/tuning_results.json"

# ---------- Model IDs ----------
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ---------- Search grid ----------
GRID = {
    "BOX_THRESHOLD":      [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "MAX_BOX_AREA_RATIO": [0.03, 0.05, 0.07, 0.1],
    "MIN_TRACK_LENGTH":   [1, 3, 5, 10],
    "TRACK_IOU_THRESHOLD": [0.2, 0.3, 0.4],
    "MAX_TRACK_GAP":      [1, 2, 3],
}

# Lowest thresholds for caching (capture all possible detections)
CACHE_BOX_THRESHOLD = min(GRID["BOX_THRESHOLD"])
CACHE_MAX_BOX_AREA_RATIO = max(GRID["MAX_BOX_AREA_RATIO"])


# ---------- Helpers ----------

def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_detections(cached, box_threshold, max_box_area_ratio, img_area):
    """Apply threshold and area filter to cached raw detections."""
    filtered = {}
    for fidx, dets in cached.items():
        boxes, names, scores = [], [], []
        for box, name, score in dets:
            if score < box_threshold:
                continue
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area / img_area > max_box_area_ratio:
                continue
            boxes.append(box)
            names.append(name)
            scores.append(score)
        filtered[fidx] = (boxes, names, scores)
    return filtered


def build_and_filter_tracks(detections, frames_list, track_iou_threshold,
                            max_track_gap, min_track_length, detect_interval):
    """Build IoU-based tracks and filter short ones. Returns per-frame filtered boxes."""
    active_tracks = []
    finished_tracks = []
    next_track_id = 0

    for frame_idx in frames_list:
        boxes, names, scores = detections.get(frame_idx, ([], [], []))
        matched_det = set()
        matched_track = set()

        if active_tracks and boxes:
            iou_pairs = []
            for ti, track in enumerate(active_tracks):
                for di, det_box in enumerate(boxes):
                    iou_val = box_iou(track["last_box"], det_box)
                    if iou_val >= track_iou_threshold:
                        iou_pairs.append((iou_val, ti, di))
            iou_pairs.sort(key=lambda x: x[0], reverse=True)
            for iou_val, ti, di in iou_pairs:
                if ti in matched_track or di in matched_det:
                    continue
                track = active_tracks[ti]
                track["entries"].append((frame_idx, boxes[di], names[di], scores[di]))
                track["last_frame"] = frame_idx
                track["last_box"] = boxes[di]
                matched_track.add(ti)
                matched_det.add(di)

        still_active = []
        for ti, track in enumerate(active_tracks):
            gap = (frame_idx - track["last_frame"]) // max(1, detect_interval)
            if ti not in matched_track and gap >= max_track_gap:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

        for di in range(len(boxes)):
            if di not in matched_det:
                active_tracks.append({
                    "id": next_track_id,
                    "entries": [(frame_idx, boxes[di], names[di], scores[di])],
                    "last_frame": frame_idx,
                    "last_box": boxes[di],
                })
                next_track_id += 1

    finished_tracks.extend(active_tracks)

    valid_tracks = [t for t in finished_tracks if len(t["entries"]) >= min_track_length]

    # Build lookup: frame_idx -> list of (box, name, score)
    result = defaultdict(list)
    for track in valid_tracks:
        for (fidx, box, name, score) in track["entries"]:
            result[fidx].append((box, name, score))

    return dict(result)


def main():
    parser = argparse.ArgumentParser(description="Tune pipeline hyperparameters")
    parser.add_argument("--metric", type=str, default="pixel_f1",
                        choices=["pixel_f1", "pixel_precision", "pixel_recall",
                                 "pixel_iou", "inst_f1", "inst_precision", "inst_recall"],
                        help="Metric to optimize (default: pixel_f1)")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5,
                        help="IoU threshold for instance matching in evaluation (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Show top K results (default: 10)")
    args = parser.parse_args()

    # --- Check labelme annotations exist ---
    json_files = sorted([f for f in os.listdir(LABELME_DIR) if f.endswith(".json")])
    if not json_files:
        print(f"No labelme annotations found in {LABELME_DIR}. Run evaluate.py --select-frames first.")
        return
    labelme_frame_ids = [os.path.splitext(f)[0] for f in json_files]
    print(f"Found {len(labelme_frame_ids)} labelme annotations: {labelme_frame_ids}")

    # --- Check frames exist ---
    frame_names = sorted([
        p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    print(f"Total frames: {num_frames}")

    frame_idx_map = {os.path.splitext(fn)[0]: i for i, fn in enumerate(frame_names)}
    sample_image = Image.open(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[0]))
    img_w, img_h = sample_image.size
    img_area = img_h * img_w

    # --- Load models ---
    print(f"\nLoading Grounding DINO from {GDINO_MODEL_ID}...")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
    print("Grounding DINO loaded.")

    print(f"Loading SAM2 from {SAM2_CHECKPOINT}...")
    sam2_image_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    print("SAM2 loaded.")

    # --- Step 1: Cache GDINO detections at lowest threshold ---
    print(f"\n--- Caching detections (threshold={CACHE_BOX_THRESHOLD}, "
          f"max_area_ratio={CACHE_MAX_BOX_AREA_RATIO}) ---")
    TEXT_PROMPT = "bubble.pore"

    cached_detections = {}  # frame_idx -> list of (box, name, score)
    for fidx in tqdm(range(num_frames), desc="Detecting"):
        fpath = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[fidx])
        image = Image.open(fpath).convert("RGB")
        inputs = gdino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gdino_model(**inputs)
        results = gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=CACHE_BOX_THRESHOLD,
            target_sizes=[(img_h, img_w)],
        )[0]

        frame_dets = []
        for bbox, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            bbox = bbox.cpu().tolist()
            score_val = round(float(score.cpu()), 4)
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if box_area / img_area > CACHE_MAX_BOX_AREA_RATIO:
                continue
            frame_dets.append((bbox, label, score_val))
        cached_detections[fidx] = frame_dets

    total_cached = sum(len(v) for v in cached_detections.values())
    print(f"Cached {total_cached} detections across {num_frames} frames")

    # --- Step 2: Pre-load labelme ground truth ---
    gt_data = {}
    for frame_id in labelme_frame_ids:
        json_path = os.path.join(LABELME_DIR, f"{frame_id}.json")
        gt_data[frame_id] = labelme_polygons_to_masks(json_path)

    # --- Step 3: Pre-encode labelme frames with SAM2 ---
    # SAM2 image features are stored per-image, so we'll encode during eval
    labelme_images = {}
    for frame_id in labelme_frame_ids:
        fpath = os.path.join(SOURCE_VIDEO_FRAME_DIR, f"{frame_id}.jpg")
        if os.path.exists(fpath):
            labelme_images[frame_id] = np.array(Image.open(fpath).convert("RGB"))

    # --- Step 4: Grid search ---
    combos = list(itertools.product(
        GRID["BOX_THRESHOLD"],
        GRID["MAX_BOX_AREA_RATIO"],
        GRID["MIN_TRACK_LENGTH"],
        GRID["TRACK_IOU_THRESHOLD"],
        GRID["MAX_TRACK_GAP"],
    ))
    print(f"\n--- Grid search: {len(combos)} parameter combinations ---")

    frames_list = list(range(num_frames))
    all_results = []

    for combo_idx, (box_thresh, max_area, min_track, track_iou, max_gap) in enumerate(
        tqdm(combos, desc="Tuning")
    ):
        # Filter cached detections by threshold and area
        filtered = filter_detections(cached_detections, box_thresh, max_area, img_area)

        # Build and filter tracks
        tracked = build_and_filter_tracks(
            filtered, frames_list, track_iou, max_gap, min_track, detect_interval=1
        )

        # Evaluate on labelme frames
        frame_metrics = []
        for frame_id in labelme_frame_ids:
            fidx = frame_idx_map.get(frame_id)
            if fidx is None:
                continue

            gt_masks = gt_data[frame_id]
            frame_boxes = tracked.get(fidx, [])

            if len(frame_boxes) == 0:
                h, w = gt_masks.shape[1], gt_masks.shape[2] if gt_masks.shape[0] > 0 else (img_h, img_w)
                pred_masks = np.zeros((0, h, w), dtype=np.uint8)
            else:
                boxes_only = [b[0] for b in frame_boxes]
                image_rgb = labelme_images.get(frame_id)
                if image_rgb is None:
                    continue
                image_predictor.set_image(image_rgb)
                boxes_np = np.array(boxes_only)
                masks, _, _ = image_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=boxes_np, multimask_output=False,
                )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                pred_masks = masks.astype(np.uint8)

            metrics = evaluate_frame(gt_masks, pred_masks, args.eval_iou_threshold)
            frame_metrics.append(metrics)

        if not frame_metrics:
            continue

        # Average metrics
        n = len(frame_metrics)
        avg_metrics = {}
        for key in frame_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in frame_metrics) / n

        result = {
            "BOX_THRESHOLD": box_thresh,
            "MAX_BOX_AREA_RATIO": max_area,
            "MIN_TRACK_LENGTH": min_track,
            "TRACK_IOU_THRESHOLD": track_iou,
            "MAX_TRACK_GAP": max_gap,
            **avg_metrics,
        }
        all_results.append(result)

    # --- Step 5: Display results ---

    # Show top-K table sorted by the user's chosen metric
    all_results.sort(key=lambda r: r[args.metric], reverse=True)

    header = (f"{'#':>3}  {'BoxTh':>6} {'MaxArea':>8} {'MinTrk':>7} {'TrkIoU':>7} {'MaxGap':>7}  "
              f"{'Px-P':>6} {'Px-R':>6} {'Px-F1':>6} {'Px-IoU':>7}  "
              f"{'In-P':>6} {'In-R':>6} {'In-F1':>6} {'mIoU':>6}")

    def format_row(i, r):
        return (f"{i:>3}  {r['BOX_THRESHOLD']:>6.2f} {r['MAX_BOX_AREA_RATIO']:>8.2f} "
                f"{r['MIN_TRACK_LENGTH']:>7d} {r['TRACK_IOU_THRESHOLD']:>7.2f} {r['MAX_TRACK_GAP']:>7d}  "
                f"{r['pixel_precision']:>6.3f} {r['pixel_recall']:>6.3f} {r['pixel_f1']:>6.3f} "
                f"{r['pixel_iou']:>7.3f}  "
                f"{r['inst_precision']:>6.3f} {r['inst_recall']:>6.3f} {r['inst_f1']:>6.3f} "
                f"{r['mean_matched_iou']:>6.3f}")

    print(f"\n{'='*120}")
    print(f"TOP {args.top_k} RESULTS (sorted by {args.metric})")
    print(f"{'='*120}")
    print(header)
    print("-" * 120)
    for i, r in enumerate(all_results[:args.top_k]):
        print(format_row(i + 1, r))

    # Show best combo for EACH key metric in one summary
    key_metrics = ["pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou",
                   "inst_precision", "inst_recall", "inst_f1"]

    def format_params(r):
        return (f"BoxTh={r['BOX_THRESHOLD']:.2f}  MaxArea={r['MAX_BOX_AREA_RATIO']:.2f}  "
                f"MinTrk={r['MIN_TRACK_LENGTH']}  TrkIoU={r['TRACK_IOU_THRESHOLD']:.2f}  "
                f"MaxGap={r['MAX_TRACK_GAP']}")

    print(f"\n{'='*120}")
    print("BEST PARAMETERS PER METRIC (single run — no need to re-run)")
    print(f"{'='*120}")
    for metric in key_metrics:
        best_for_metric = max(all_results, key=lambda r: r[metric])
        print(f"\n  {metric} = {best_for_metric[metric]:.4f}")
        print(f"    {format_params(best_for_metric)}")
        print(f"    Px: P={best_for_metric['pixel_precision']:.3f}  R={best_for_metric['pixel_recall']:.3f}  "
              f"F1={best_for_metric['pixel_f1']:.3f}  IoU={best_for_metric['pixel_iou']:.3f}  |  "
              f"In: P={best_for_metric['inst_precision']:.3f}  R={best_for_metric['inst_recall']:.3f}  "
              f"F1={best_for_metric['inst_f1']:.3f}")

    # Save full results
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results ({len(all_results)} combos) saved to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
