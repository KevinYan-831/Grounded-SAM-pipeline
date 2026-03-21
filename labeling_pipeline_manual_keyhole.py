#!/usr/bin/env python3
"""
Manual-keyhole labeling pipeline.

This variant uses LabelMe point annotations to track keyhole:
  1. User labels keyhole start/end frames as points in LabelMe.
  2. Prompt points are linearly interpolated between start/end frames.
  3. SAM2 point-prompt segmentation is run per frame in that range.
  4. Bubble analysis is limited to the annotated frame range.

LabelMe annotations expected in --labelme-dir (default: --frames-dir):
  - Preferred labels: keyhole_start, keyhole_end (shape_type=point)
  - Fallback label: keyhole on two frames (earliest=start, latest=end)
"""

import argparse
import contextlib
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

from labeling_pipeline import (
    CROP_TOP,
    DETECTION_CACHE_PATH,
    FRAME_EXTS,
    GDINO_MODEL_ID,
    OUTPUT_FRAMES_DIR,
    OUTPUT_JSON_PATH,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CFG,
    SOURCE_VIDEO_FRAME_DIR,
    VIDEO_PATH,
    classify_bubble_tracks,
    generate_labeled_frames,
    group_into_intervals,
    label_frames,
    list_frame_files,
    load_detection_cache,
    merge_fragmented_tracks,
    save_detection_cache,
)
from utils.detection_utils import build_and_filter_tracks, detect_on_frame, extract_and_crop_frames, load_models
from utils.keyhole_detector import _smooth_positions


def _parse_frame_index_from_stem(stem):
    try:
        return int(stem)
    except ValueError:
        nums = re.findall(r"\d+", stem)
        if nums:
            return int(nums[-1])
        return None


def _extract_first_point(shape):
    points = shape.get("points") or []
    if not points:
        return None
    pt = points[0]
    if len(pt) < 2:
        return None
    return [float(pt[0]), float(pt[1])]


def load_manual_keyhole_annotation(
    labelme_dir,
    start_label="keyhole_start",
    end_label="keyhole_end",
    fallback_label="keyhole",
):
    """
    Read LabelMe JSON files and extract manual keyhole start/end annotations.

    Returns:
        dict(start_frame, end_frame, start_point_xy, end_point_xy, sources)
    """
    json_files = sorted(
        [
            os.path.join(labelme_dir, p)
            for p in os.listdir(labelme_dir)
            if p.lower().endswith(".json")
        ]
    )

    if not json_files:
        raise FileNotFoundError(
            f"No LabelMe JSON files found in {labelme_dir}. "
            "Please annotate keyhole start/end points first."
        )

    start_label = start_label.lower()
    end_label = end_label.lower()
    fallback_label = fallback_label.lower()

    start_candidates = []
    end_candidates = []
    generic_candidates = []

    for jpath in json_files:
        stem = os.path.splitext(os.path.basename(jpath))[0]
        fidx = _parse_frame_index_from_stem(stem)
        if fidx is None:
            continue

        with open(jpath, "r") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "point":
                continue
            point = _extract_first_point(shape)
            if point is None:
                continue

            label = str(shape.get("label", "")).strip().lower()
            item = {
                "frame": fidx,
                "point": point,
                "json_file": jpath,
                "label": label,
            }

            if label == start_label:
                start_candidates.append(item)
            elif label == end_label:
                end_candidates.append(item)
            elif label == fallback_label:
                generic_candidates.append(item)

    if start_candidates and end_candidates:
        start = sorted(start_candidates, key=lambda x: x["frame"])[0]
        end = sorted(end_candidates, key=lambda x: x["frame"])[-1]
    else:
        if len(generic_candidates) < 2:
            raise ValueError(
                "Could not infer manual keyhole start/end from LabelMe annotations. "
                f"Expected either labels '{start_label}' and '{end_label}', "
                f"or at least two '{fallback_label}' point annotations."
            )
        ordered = sorted(generic_candidates, key=lambda x: x["frame"])
        start = ordered[0]
        end = ordered[-1]

    if start["frame"] > end["frame"]:
        start, end = end, start

    return {
        "start_frame": int(start["frame"]),
        "end_frame": int(end["frame"]),
        "start_point_xy": [float(start["point"][0]), float(start["point"][1])],
        "end_point_xy": [float(end["point"][0]), float(end["point"][1])],
        "sources": [start["json_file"], end["json_file"]],
    }


def _interp_point(start_frame, end_frame, start_xy, end_xy, fidx):
    if end_frame <= start_frame:
        return float(start_xy[0]), float(start_xy[1])
    t = (fidx - start_frame) / float(end_frame - start_frame)
    x = start_xy[0] + t * (end_xy[0] - start_xy[0])
    y = start_xy[1] + t * (end_xy[1] - start_xy[1])
    return float(x), float(y)


def _bbox_from_mask(mask):
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return None
    x1 = float(x_idx.min())
    y1 = float(y_idx.min())
    x2 = float(x_idx.max())
    y2 = float(y_idx.max())
    return [x1, y1, x2, y2]


def _bbox_valid_for_keyhole(bbox, config):
    """
    Validate SAM-derived keyhole bbox.

    The keyhole is a tall, narrow vertical feature. Enforce max_width and
    min_aspect_ratio from config to reject SAM masks that segment the
    keyhole + nearby bubbles together as one big region.
    """
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    kh_cfg = config.get("detection", {}).get("keyhole", {})
    max_w = kh_cfg.get("max_width", float("inf"))
    min_ar = kh_cfg.get("min_aspect_ratio", 0.0)

    ar = h / w if w > 0 else 0.0
    return w <= max_w and ar >= min_ar


def _make_point_box(px, py, img_w, img_h, half_size):
    x1 = max(0.0, px - half_size)
    y1 = max(0.0, py - half_size)
    x2 = min(float(img_w - 1), px + half_size)
    y2 = min(float(img_h - 1), py + half_size)
    return [x1, y1, x2, y2]


def _sam_predict_with_point(image_predictor, img_np, point_xy, device):
    """
    Run SAM2 point-prompt and return all masks sorted by score (highest first).
    """
    import torch

    point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    if str(device).startswith("cuda"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.inference_mode(), autocast_ctx:
        image_predictor.set_image(img_np)
        masks, scores, _ = image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=True,
        )

    # Return masks sorted by score, highest first
    order = np.argsort(scores)[::-1]
    return [masks[i] for i in order]


def build_manual_keyhole_positions(
    frame_names,
    source_dir,
    manual_ann,
    image_predictor,
    device,
    config,
):
    """
    Build keyhole trajectory by interpolating LabelMe prompt points and
    running SAM point-prompt segmentation on each frame in [start, end].
    """
    manual_cfg = config.get("manual_keyhole", {})
    fallback_half = float(manual_cfg.get("fallback_box_half_size_px", 10))
    smoothing_window = int(manual_cfg.get("smoothing_window", 3))
    keep_prev_on_invalid = bool(manual_cfg.get("keep_previous_bbox_on_invalid_mask", True))

    start_frame = int(manual_ann["start_frame"])
    end_frame = int(manual_ann["end_frame"])
    start_xy = manual_ann["start_point_xy"]
    end_xy = manual_ann["end_point_xy"]

    positions = OrderedDict()
    prev_bbox = None
    fallback_count = 0

    for fidx in tqdm(range(start_frame, end_frame + 1), desc="Manual keyhole SAM"):
        fpath = os.path.join(source_dir, frame_names[fidx])
        img_np = np.array(PILImage.open(fpath).convert("RGB"))
        img_h, img_w = img_np.shape[:2]

        px, py = _interp_point(start_frame, end_frame, start_xy, end_xy, fidx)

        bbox = None
        try:
            masks = _sam_predict_with_point(
                image_predictor=image_predictor,
                img_np=img_np,
                point_xy=(px, py),
                device=device,
            )
            # Try each mask (highest score first), pick first valid one
            for mask in masks:
                raw_bbox = _bbox_from_mask(mask)
                if _bbox_valid_for_keyhole(raw_bbox, config):
                    bbox = raw_bbox
                    break
        except Exception:
            bbox = None

        if bbox is None:
            if keep_prev_on_invalid and prev_bbox is not None:
                bbox = prev_bbox
            else:
                bbox = _make_point_box(px, py, img_w, img_h, fallback_half)
            fallback_count += 1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        positions[fidx] = {
            "cx": cx,
            "cy": cy,
            "bbox": bbox,
            "prompt_point": [px, py],
        }
        prev_bbox = bbox

    positions = _smooth_positions(positions, smoothing_window)

    if fallback_count:
        print(
            f"  Manual keyhole SAM fallback used on {fallback_count} frames "
            "(invalid/failed mask)."
        )

    return dict(positions)


def run_bubble_detections_subset(
    frame_names,
    source_dir,
    frame_indices,
    config,
    gdino_processor,
    gdino_model,
    img_h,
    img_w,
    device,
):
    """Run GDINO bubble detections only on selected frame indices."""
    bb_cfg = config["detection"]["bubble"]
    text_prompt = bb_cfg["text_prompt"]
    detection_threshold = bb_cfg["box_threshold"]
    detection_area_ratio = bb_cfg["max_box_area_ratio"]

    all_detections = {}
    for fidx in tqdm(frame_indices, desc="Bubble detection"):
        boxes, names, scores = detect_on_frame(
            fidx,
            frame_names,
            source_dir,
            gdino_processor,
            gdino_model,
            text_prompt,
            detection_threshold,
            img_h,
            img_w,
            detection_area_ratio,
            device,
        )
        all_detections[fidx] = (boxes, names, scores)

    total = sum(len(v[0]) for v in all_detections.values())
    frames_with = sum(1 for v in all_detections.values() if len(v[0]) > 0)
    print(f"  Total bubble detections: {total} across {frames_with} analyzed frames")
    return all_detections


def filter_bubble_detections_with_keyhole(
    all_detections,
    keyhole_positions,
    config,
    img_h,
    img_w,
):
    """
    Filter bubble detections:
      1. Remove boxes that overlap the keyhole bbox.
      2. Remove boxes whose center is to the LEFT of the keyhole center
         (only bubbles to the right of the keyhole are real).
    """
    bb_cfg = config["detection"]["bubble"]
    bb_threshold = bb_cfg["box_threshold"]
    bb_max_area = bb_cfg["max_box_area_ratio"]
    img_area = img_h * img_w

    filtered = {}
    removed_overlap = 0
    removed_left = 0

    for fidx, det in all_detections.items():
        boxes, names, scores = det
        keep_boxes, keep_names, keep_scores = [], [], []

        kh = keyhole_positions.get(fidx, {})
        kh_box = kh.get("bbox")
        kh_cx = kh.get("cx")

        for box, name, score in zip(boxes, names, scores):
            if score < bb_threshold:
                continue
            x1, y1, x2, y2 = box
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(img_area)
            if area_ratio > bb_max_area:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Remove bubbles overlapping keyhole
            if kh_box is not None:
                if kh_box[0] <= cx <= kh_box[2] and kh_box[1] <= cy <= kh_box[3]:
                    removed_overlap += 1
                    continue

            # Only keep bubbles to the RIGHT of the keyhole
            if kh_cx is not None and cx <= kh_cx:
                removed_left += 1
                continue

            keep_boxes.append(box)
            keep_names.append(name)
            keep_scores.append(score)

        filtered[fidx] = (keep_boxes, keep_names, keep_scores)

    if removed_overlap:
        print(f"  Removed {removed_overlap} bubble boxes overlapping keyhole")
    if removed_left:
        print(f"  Removed {removed_left} bubble boxes to the left of keyhole")
    return filtered


def load_sam_only(sam2_checkpoint, sam2_model_cfg):
    """Load only SAM2 (used when --skip-detection is enabled)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAM2 from {sam2_checkpoint}...")
    sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    print("SAM2 loaded.")

    return image_predictor, device


def _load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Manual keyhole labeling pipeline (LabelMe + SAM interpolation)"
    )
    parser.add_argument(
        "--config", type=str, default="labeling_rules.yaml",
        help="Labeling rules YAML (default: labeling_rules.yaml)",
    )
    parser.add_argument(
        "--video-path", type=str, default=VIDEO_PATH,
        help=f"Input video path (default: {VIDEO_PATH})",
    )
    parser.add_argument(
        "--frames-dir", type=str, default=SOURCE_VIDEO_FRAME_DIR,
        help=f"Frames directory (default: {SOURCE_VIDEO_FRAME_DIR})",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip frame extraction (reuse already-extracted frames)",
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Skip GDINO detection and reuse cached raw_detections.json",
    )
    parser.add_argument(
        "--detection-cache", type=str, default=DETECTION_CACHE_PATH,
        help=f"Detection cache path (default: {DETECTION_CACHE_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_JSON_PATH,
        help=f"Output JSON path (default: {OUTPUT_JSON_PATH})",
    )
    parser.add_argument(
        "--output-frames-dir", type=str, default=OUTPUT_FRAMES_DIR,
        help=f"Output frames directory (default: {OUTPUT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--labelme-dir", type=str, default=None,
        help="Directory containing LabelMe JSONs for this trajectory (default: frames-dir)",
    )
    parser.add_argument(
        "--manual-start-label", type=str, default="keyhole_start",
        help="LabelMe point label for keyhole start frame (default: keyhole_start)",
    )
    parser.add_argument(
        "--manual-end-label", type=str, default="keyhole_end",
        help="LabelMe point label for keyhole end frame (default: keyhole_end)",
    )
    parser.add_argument(
        "--manual-fallback-label", type=str, default="keyhole",
        help="Fallback LabelMe point label when start/end labels are not used (default: keyhole)",
    )
    args = parser.parse_args()

    video_path = args.video_path
    frames_dir = args.frames_dir
    detection_cache_path = args.detection_cache
    output_frames_dir = args.output_frames_dir

    config = _load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Frames
    if args.skip_extraction:
        frame_names = list_frame_files(frames_dir)
        if not frame_names:
            raise FileNotFoundError(
                f"No frames found in {frames_dir}. Expected extensions: {FRAME_EXTS}"
            )
        sample = PILImage.open(os.path.join(frames_dir, frame_names[0]))
        img_w, img_h = sample.size
        print(f"Reusing {len(frame_names)} frames from {frames_dir}")
    else:
        frame_names, img_w, img_h = extract_and_crop_frames(
            video_path,
            frames_dir,
            crop_top=CROP_TOP,
        )

    num_frames = len(frame_names)
    print(f"Total frames: {num_frames}, image size: {img_w}x{img_h}")

    labelme_dir = args.labelme_dir or frames_dir
    manual_ann = load_manual_keyhole_annotation(
        labelme_dir=labelme_dir,
        start_label=args.manual_start_label,
        end_label=args.manual_end_label,
        fallback_label=args.manual_fallback_label,
    )

    start_frame = manual_ann["start_frame"]
    end_frame = manual_ann["end_frame"]

    if start_frame < 0 or end_frame >= num_frames:
        raise ValueError(
            f"Manual annotation frame range [{start_frame}, {end_frame}] is out of bounds "
            f"for {num_frames} frames."
        )

    analyzed_frames = list(range(start_frame, end_frame + 1))
    print(
        f"Manual keyhole annotation loaded: frames {start_frame} to {end_frame} "
        f"({len(analyzed_frames)} analyzed frames)"
    )

    # Models
    if args.skip_detection:
        gdino_processor = gdino_model = None
        image_predictor, device = load_sam_only(SAM2_CHECKPOINT, SAM2_MODEL_CFG)
    else:
        gdino_processor, gdino_model, image_predictor, device = load_models(
            GDINO_MODEL_ID,
            SAM2_CHECKPOINT,
            SAM2_MODEL_CFG,
        )

    # Bubble detections
    print("\n--- Bubble detection ---")
    if args.skip_detection:
        print(f"  Loading from cache: {detection_cache_path}")
        all_detections = load_detection_cache(detection_cache_path)
        all_detections = {f: all_detections.get(f, ([], [], [])) for f in analyzed_frames}
    else:
        all_detections = run_bubble_detections_subset(
            frame_names=frame_names,
            source_dir=frames_dir,
            frame_indices=analyzed_frames,
            config=config,
            gdino_processor=gdino_processor,
            gdino_model=gdino_model,
            img_h=img_h,
            img_w=img_w,
            device=device,
        )
        save_detection_cache(all_detections, detection_cache_path)

    # Manual keyhole trajectory
    print("\n--- Building manual keyhole trajectory (LabelMe + SAM2) ---")
    keyhole_positions = build_manual_keyhole_positions(
        frame_names=frame_names,
        source_dir=frames_dir,
        manual_ann=manual_ann,
        image_predictor=image_predictor,
        device=device,
        config=config,
    )
    first_keyhole_frame = start_frame

    # Filter bubbles against keyhole overlap
    bubble_detections = filter_bubble_detections_with_keyhole(
        all_detections=all_detections,
        keyhole_positions=keyhole_positions,
        config=config,
        img_h=img_h,
        img_w=img_w,
    )

    # Build bubble tracks only on analyzed frames
    print("\n--- Building bubble tracks ---")
    track_cfg = config["tracking"]
    bubble_tracks, _ = build_and_filter_tracks(
        bubble_detections,
        analyzed_frames,
        track_iou_threshold=track_cfg["track_iou_threshold"],
        max_track_gap=track_cfg["max_track_gap"],
        min_track_length=track_cfg["bubble_min_track_length"],
    )
    print(f"  Bubble tracks after filtering: {len(bubble_tracks)}")

    print("\n--- Merging fragmented bubble tracks ---")
    bubble_tracks = merge_fragmented_tracks(bubble_tracks, config)

    print("\n--- Classifying bubble tracks ---")
    classified_tracks = classify_bubble_tracks(
        bubble_tracks,
        keyhole_positions,
        config,
        end_frame=end_frame,
    )

    # Labels over analyzed range only (generation-based)
    print("\n--- Labeling frames ---")
    frame_labels = label_frames(
        classified_tracks,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    # NOTE: No label smoothing for generation-based labels.
    # Birth events are single-frame; majority-vote smoothing erases them.
    intervals = group_into_intervals(frame_labels, config)

    labels_cfg = config["labels"]
    label_name_map = {v["id"]: v["name"] for v in labels_cfg.values()}
    count_by_label = Counter(frame_labels.values())

    print("\n" + "=" * 60)
    print("LABELING SUMMARY (MANUAL KEYHOLE)")
    print("=" * 60)
    print(f"Keyhole source:          LabelMe points + SAM2")
    print(f"Manual keyhole range:    {start_frame} - {end_frame}")
    print(f"Total frames:            {num_frames}")
    print(f"Analyzed frames:         {len(analyzed_frames)}")
    print(f"Total intervals:         {len(intervals)}")
    print()
    for label_id in sorted(count_by_label.keys()):
        name = label_name_map.get(label_id, "Unknown")
        count = count_by_label[label_id]
        pct = 100.0 * count / len(analyzed_frames)
        print(f"  [{label_id}] {name}: {count} frames ({pct:.1f}%)")

    # Labeled frame visualization
    print("\n--- Generating labeled frames ---")
    generate_labeled_frames(
        frame_names,
        frames_dir,
        output_frames_dir,
        frame_labels,
        keyhole_positions,
        classified_tracks,
        config,
    )

    # Output JSON
    frame_track_lookup = defaultdict(list)
    for track in classified_tracks:
        for entry in track["entries"]:
            frame_track_lookup[entry[0]].append(track)

    frame_label_details = []
    for fidx in analyzed_frames:
        label_id = frame_labels.get(fidx, 0)
        tracks_here = frame_track_lookup.get(fidx, [])

        detail = {
            "frame_index": fidx,
            "frame_file": frame_names[fidx] if fidx < len(frame_names) else None,
            "label_id": label_id,
            "label_name": label_name_map.get(label_id, "Unknown"),
            "keyhole_present": fidx in keyhole_positions,
            "num_bubbles": len(tracks_here),
            "bubble_details": [
                {
                    "track_id": t["id"],
                    "near_keyhole": t["is_near_keyhole"],
                    "permanent": t["is_permanent"],
                    "distance_to_keyhole": t["avg_distance_to_keyhole"],
                }
                for t in tracks_here
            ],
        }
        if fidx in keyhole_positions:
            kh = keyhole_positions[fidx]
            detail["keyhole_center"] = [round(kh["cx"], 2), round(kh["cy"], 2)]
            detail["keyhole_bbox"] = [round(v, 2) for v in kh["bbox"]]
        frame_label_details.append(detail)

    # Build compact label sequence and transition counts for the analyzed range
    label_sequence = [frame_labels.get(fidx, 0) for fidx in analyzed_frames]
    analyzed_label_sequence = label_sequence

    num_labels = 4
    transition_counts = [[0] * num_labels for _ in range(num_labels)]
    for i in range(len(analyzed_label_sequence) - 1):
        src = analyzed_label_sequence[i]
        dst = analyzed_label_sequence[i + 1]
        if 0 <= src < num_labels and 0 <= dst < num_labels:
            transition_counts[src][dst] += 1

    output = {
        "video_path": video_path,
        "config_file": args.config,
        "keyhole_detection_method": "manual_labelme_sam",
        "manual_keyhole_annotation": manual_ann,
        "first_keyhole_frame": first_keyhole_frame,
        "last_keyhole_frame": end_frame,
        "analysis_frame_range": [start_frame, end_frame],
        "image_size": {"width": img_w, "height": img_h},
        "total_frames": num_frames,
        "label_definitions": {
            k: {"id": v["id"], "name": v["name"], "description": v["description"]}
            for k, v in labels_cfg.items()
        },
        "summary": {
            label_name_map.get(lid, "Unknown"): count_by_label.get(lid, 0)
            for lid in range(4)
        },
        "intervals": intervals,
        "tracks": {
            "bubble_tracks": [
                {
                    "track_id": t["id"],
                    "duration": t["duration"],
                    "is_near_keyhole": t["is_near_keyhole"],
                    "is_permanent": t["is_permanent"],
                    "avg_distance_to_keyhole": t["avg_distance_to_keyhole"],
                    "first_frame": t["entries"][0][0],
                    "last_frame": t["entries"][-1][0],
                }
                for t in classified_tracks
            ]
        },
        "label_sequence": label_sequence,
        "analyzed_label_sequence": analyzed_label_sequence,
        "transition_counts": transition_counts,
        "frame_labels": frame_label_details,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nLabeling results saved to {args.output}")


if __name__ == "__main__":
    main()
