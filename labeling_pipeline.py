"""
Frame labeling pipeline for bubble detection in X-ray video.

Assigns per-frame state labels based on keyhole presence, bubble proximity
to the keyhole, and track persistence.  Outputs a labeled interval dataset
as JSON.

Keyhole detection uses Grounding DINO ("bubble.pore" prompt) — the same
model used for bubbles.  Detections are split by shape:
  • leftmost detection with h ≥ min_height                  → keyhole
  • everything else                                          → bubble

Labels:
  0 - No Signal:                      No keyhole detected
  1 - Normal Process:                 Keyhole present, NO bubbles in entire video
  2 - Unstable without Pore:          Keyhole present, no bubble this frame, but
                                      bubbles exist elsewhere in the video
  3 - Transient Pore Generation:      Bubble at this frame that will disappear
  4 - Permanent Pore Generation:      Bubble at this frame that stays permanently

Usage:
    python labeling_pipeline.py
    python labeling_pipeline.py --config custom_rules.yaml
    python labeling_pipeline.py --skip-extraction   # reuse already-extracted frames
    python labeling_pipeline.py --skip-detection    # reuse cached detections JSON
"""

import os
import json
import yaml
import argparse
import re
import cv2
import numpy as np

from collections import Counter, defaultdict
from tqdm import tqdm

from utils.detection_utils import (
    load_models,
    extract_and_crop_frames,
    detect_on_frame,
    build_and_filter_tracks,
    box_iou,
    read_image_any,
)
from utils.keyhole_detector import (
    _temporal_filter,
    _find_first_keyhole_frame,
    _interpolate_positions,
    _smooth_positions,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = "./data/raw/x_ray_video.mp4"
SOURCE_VIDEO_FRAME_DIR = "./data/frames/custom_video_frames"
OUTPUT_JSON_PATH = "./data/labeling/labeling_results.json"
OUTPUT_FRAMES_DIR = "./data/labeling/frames"
DETECTION_CACHE_PATH = "./data/labeling/raw_detections.json"
CROP_TOP = 800 - 310  # keep bottom 310 rows
CROP_BOTTOM_HEIGHT = 310
FRAME_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ─────────────────────────────────────────────────────────────
# Detection: run GDINO once per frame with wide thresholds
# ─────────────────────────────────────────────────────────────
def run_all_detections(
    frame_names, source_dir, config, gdino_processor, gdino_model,
    img_h, img_w, device, crop_bottom_height=None,
):
    """
    Run Grounding DINO on every frame using parameters broad enough to
    capture both bubbles and keyhole detections.

    We run with the LOWER of the two thresholds and the HIGHER of the two
    area ratios so that keyhole candidates (lower confidence, larger box)
    are not filtered out before we can classify them by shape.

    Returns:
        all_detections: {frame_idx: (boxes, names, scores)}
    """
    bb_cfg = config["detection"]["bubble"]
    kh_cfg = config["detection"]["keyhole"]
    sel_cfg = config.get("keyhole_selection", {})

    text_prompt = bb_cfg["text_prompt"]
    detection_threshold = min(
        bb_cfg["box_threshold"],
        kh_cfg["box_threshold"],
        float(sel_cfg.get("track_box_threshold", kh_cfg["box_threshold"])),
        float(kh_cfg.get("rescue_box_threshold", kh_cfg["box_threshold"])),
    )
    detection_area_ratio = max(bb_cfg["max_box_area_ratio"], kh_cfg["max_box_area_ratio"])

    all_detections = {}
    for fidx in tqdm(range(len(frame_names)), desc="Detection"):
        boxes, names, scores = detect_on_frame(
            fidx, frame_names, source_dir, gdino_processor, gdino_model,
            text_prompt, detection_threshold,
            img_h, img_w, detection_area_ratio, device,
            crop_bottom_height=crop_bottom_height,
        )
        all_detections[fidx] = (boxes, names, scores)

    total = sum(len(v[0]) for v in all_detections.values())
    frames_with = sum(1 for v in all_detections.values() if len(v[0]) > 0)
    print(f"  Total raw detections: {total} across {frames_with} frames")
    return all_detections


# ─────────────────────────────────────────────────────────────
# Split detections: leftmost tall object → keyhole, rest → bubble
# ─────────────────────────────────────────────────────────────
def split_keyhole_and_bubbles(all_detections, config, img_h, img_w, keyhole_raw_override=None):
    """
    Partition GDINO detections into keyhole candidates and bubbles.

    Strategy: the keyhole sits at the leading edge of the weld pool and
    moves right→left over time.  In any given frame, the LEFTMOST detection
    that passes ALL shape constraints is the keyhole candidate.  All other
    detections that pass the bubble threshold and area filter are bubbles.

    Shape constraints for keyhole candidates:
      • height  >= min_height      (keyhole is tall)
      • width   <= max_width       (keyhole is narrow — rejects huge boxes)
      • h/w     >= min_aspect_ratio (taller than wide — rejects wide boxes)
      • y1      <= max_top_y       (keyhole extends from image top)
      • score   >= box_threshold

    Returns:
        keyhole_raw: {frame_idx: dict(x, y, w, h, cx, cy, score)}
        bubble_detections: {frame_idx: (boxes, names, scores)}
    """
    bb_cfg = config["detection"]["bubble"]
    kh_cfg = config["detection"]["keyhole"]
    img_area = img_h * img_w

    min_h     = kh_cfg["min_height"]
    max_w     = kh_cfg.get("max_width", float("inf"))
    min_ar    = kh_cfg.get("min_aspect_ratio", 0.0)
    max_top_y = kh_cfg.get("max_top_y", float("inf"))
    kh_threshold = kh_cfg["box_threshold"]
    bb_threshold = bb_cfg["box_threshold"]
    bb_max_area  = bb_cfg["max_box_area_ratio"]
    sel_cfg = config.get("keyhole_selection", {})
    relaxed_shape_enable = bool(sel_cfg.get("enable_shape_relaxation", True))
    relaxed_threshold = float(kh_cfg.get("rescue_box_threshold", max(0.01, kh_threshold * 0.6)))
    relaxed_min_h = float(sel_cfg.get("relaxed_min_height_px", max(6.0, min_h * 0.35)))
    relaxed_min_ar = float(sel_cfg.get("relaxed_min_aspect_ratio", max(0.15, min_ar * 0.3)))
    relaxed_max_w = sel_cfg.get("relaxed_max_width_px", None)
    if relaxed_max_w is None:
        if max_w == float("inf"):
            relaxed_max_w = float("inf")
        else:
            relaxed_max_w = float(max_w) * 2.8
    else:
        relaxed_max_w = float(relaxed_max_w)
    relaxed_top_y = float(sel_cfg.get("relaxed_max_top_y", max_top_y + sel_cfg.get("relaxed_top_y_slack_px", 140.0)))
    bubble_exclusion_iou = float(sel_cfg.get("bubble_exclusion_iou", 0.6))
    bubble_exclusion_center_px = float(sel_cfg.get("bubble_exclusion_center_px", 10.0))
    bubble_exclusion_area_ratio_min = float(sel_cfg.get("bubble_exclusion_area_ratio_min", 0.5))
    bubble_exclusion_area_ratio_max = float(sel_cfg.get("bubble_exclusion_area_ratio_max", 2.0))
    bubble_exclusion_overlap_min = float(sel_cfg.get("bubble_exclusion_overlap_min", 0.6))

    keyhole_raw       = {} if keyhole_raw_override is None else dict(keyhole_raw_override)
    bubble_detections = {}

    for fidx, (boxes, names, scores) in all_detections.items():
        # --- find keyhole (shape-based) unless override is provided ---
        kh_best = keyhole_raw.get(fidx) if keyhole_raw_override is not None else None
        if keyhole_raw_override is None:
            kh_best_x1 = float("inf")
            for box, name, score in zip(boxes, names, scores):
                x1, y1, x2, y2 = box
                h_px = y2 - y1
                w_px = x2 - x1
                ar   = h_px / w_px if w_px > 0 else 0.0

                if (w_px > 1.0 and h_px > 1.0 and
                        h_px >= min_h and
                        w_px <= max_w and
                        ar   >= min_ar and
                        y1   <= max_top_y and
                        score >= kh_threshold):
                    if x1 < kh_best_x1:
                        kh_best_x1 = x1
                        kh_best = {
                            "x": x1, "y": y1, "w": w_px, "h": h_px,
                            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                            "score": float(score),
                        }

            # Rescue for tiny/dim keyholes when strict shape misses.
            if kh_best is None and relaxed_shape_enable:
                kh_rescue_x1 = float("inf")
                for box, name, score in zip(boxes, names, scores):
                    x1, y1, x2, y2 = box
                    h_px = y2 - y1
                    w_px = x2 - x1
                    ar = h_px / w_px if w_px > 0 else 0.0
                    if (w_px > 1.0 and h_px > 1.0 and
                            h_px >= relaxed_min_h and
                            w_px <= relaxed_max_w and
                            ar >= relaxed_min_ar and
                            y1 <= relaxed_top_y and
                            score >= relaxed_threshold):
                        if x1 < kh_rescue_x1:
                            kh_rescue_x1 = x1
                            kh_best = {
                                "x": x1, "y": y1, "w": w_px, "h": h_px,
                                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                                "score": float(score),
                            }

        # --- bubbles: everything that is NOT the keyhole box ---
        bb_boxes, bb_names, bb_scores = [], [], []
        kh_box = None
        if kh_best is not None:
            kh_box = [
                kh_best["x"], kh_best["y"],
                kh_best["x"] + kh_best["w"], kh_best["y"] + kh_best["h"],
            ]
            kh_cx = kh_best["cx"]
            kh_cy = kh_best["cy"]
            kh_area = max(1.0, kh_best["w"] * kh_best["h"])

        for box, name, score in zip(boxes, names, scores):
            x1, y1, x2, y2 = box
            # Skip boxes that are the same object as keyhole.
            if kh_box is not None:
                iou = box_iou(box, kh_box)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center_dist = float(np.hypot(cx - kh_cx, cy - kh_cy))
                box_area = max(1.0, (x2 - x1) * (y2 - y1))
                area_ratio = box_area / kh_area
                similar_scale = (
                    bubble_exclusion_area_ratio_min <= area_ratio <= bubble_exclusion_area_ratio_max
                )
                inter_w = max(0.0, min(x2, kh_box[2]) - max(x1, kh_box[0]))
                inter_h = max(0.0, min(y2, kh_box[3]) - max(y1, kh_box[1]))
                inter_area = inter_w * inter_h
                overlap_min = inter_area / max(1.0, min(box_area, kh_area))
                if (
                    iou >= bubble_exclusion_iou
                    or overlap_min >= bubble_exclusion_overlap_min
                    or (center_dist <= bubble_exclusion_center_px and similar_scale)
                ):
                    continue
            if score >= bb_threshold:
                h_px = y2 - y1
                w_px = x2 - x1
                if (h_px * w_px) / img_area <= bb_max_area:
                    bb_boxes.append(box)
                    bb_names.append(name)
                    bb_scores.append(score)

        if kh_best is not None and keyhole_raw_override is None:
            keyhole_raw[fidx] = kh_best
        bubble_detections[fidx] = (bb_boxes, bb_names, bb_scores)

    kh_total = len(keyhole_raw)
    bb_total = sum(len(v[0]) for v in bubble_detections.values())
    print(f"  Keyhole raw detections: {kh_total} frames")
    print(f"  Bubble detections:      {bb_total} total across "
          f"{sum(1 for v in bubble_detections.values() if len(v[0]) > 0)} frames")
    return keyhole_raw, bubble_detections


# ─────────────────────────────────────────────────────────────
# Optional: SAM2 point-prompt refinement of keyhole bboxes
# ─────────────────────────────────────────────────────────────
def refine_keyhole_with_sam(
    keyhole_raw, frame_names, source_dir, image_predictor, device, config,
    crop_bottom_height=None,
):
    """
    For each frame in keyhole_raw, pass the approximate keyhole center
    as a point prompt to SAM2 and replace the bounding box with the one
    derived from the SAM mask.  This gives more precise keyhole bounds.

    The SAM output is validated against the same shape constraints used
    during detection.  If SAM produces an oversized or malformed box
    (e.g. full-screen segment), the original GDINO box is kept instead.

    Args:
        keyhole_raw:     {frame_idx: dict(x, y, w, h, cx, cy, score)}
        frame_names:     sorted list of frame filenames
        source_dir:      directory containing frames
        image_predictor: SAM2ImagePredictor instance
        device:          torch device string
        config:          full labeling_rules config dict (for shape constraints)

    Returns:
        Refined keyhole_raw dict (same keys, updated bbox entries).
    """
    import torch

    kh_cfg    = config["detection"]["keyhole"]
    min_h     = kh_cfg["min_height"]
    max_w     = kh_cfg.get("max_width", float("inf"))
    min_ar    = kh_cfg.get("min_aspect_ratio", 0.0)
    max_top_y = kh_cfg.get("max_top_y", float("inf"))

    fallback_count = 0
    refined = {}

    for fidx in tqdm(sorted(keyhole_raw.keys()), desc="SAM keyhole refinement"):
        kh    = keyhole_raw[fidx]
        fpath = os.path.join(source_dir, frame_names[fidx])
        img_np = read_image_any(fpath)
        if crop_bottom_height is not None and img_np.shape[0] > crop_bottom_height:
            img_np = img_np[-crop_bottom_height:, :, :]

        cx, cy = kh["cx"], kh["cy"]
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1])  # 1 = foreground

        try:
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                image_predictor.set_image(img_np)
                masks, mask_scores, _ = image_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                    normalize_coords=True,
                )

            # Pick highest-scoring mask
            best_idx = int(np.argmax(mask_scores))
            mask     = masks[best_idx]  # (H, W) bool array

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0:
                refined[fidx] = kh
                continue

            x1 = float(x_idx.min())
            y1 = float(y_idx.min())
            x2 = float(x_idx.max())
            y2 = float(y_idx.max())
            w_sam = x2 - x1
            h_sam = y2 - y1
            ar_sam = h_sam / w_sam if w_sam > 0 else 0.0

            # Validate SAM bbox with the same shape constraints as detection.
            # If SAM segments a huge region (e.g. full screen), fall back
            # to the original GDINO box which already passed these checks.
            if (h_sam >= min_h and
                    w_sam <= max_w and
                    ar_sam >= min_ar and
                    y1 <= max_top_y):
                refined[fidx] = {
                    "x": x1, "y": y1,
                    "w": w_sam, "h": h_sam,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "score": kh["score"],
                }
            else:
                refined[fidx] = kh  # SAM box too large — keep GDINO bbox
                fallback_count += 1

        except Exception as e:
            print(f"  SAM failed on frame {fidx}: {e} — keeping GDINO bbox")
            refined[fidx] = kh
            fallback_count += 1

    if fallback_count:
        print(f"  SAM refinement: kept GDINO bbox on {fallback_count} frames "
              f"(SAM output failed shape constraints)")
    return refined


# ─────────────────────────────────────────────────────────────
# Merge fragmented bubble tracks
# ─────────────────────────────────────────────────────────────
def merge_fragmented_tracks(tracks, config):
    """
    Merge tracks that belong to the same physical bubble but were split
    because GDINO missed detections for several frames.

    Two tracks are merged when:
      • The time gap between them is <= max_gap_frames
      • The spatial distance between the last box of the earlier track and
        the first box of the later track is <= max_distance_px

    Merged tracks inherit the ID of the earlier track.  The combined entry
    list is sorted by frame index so downstream code sees a single track.

    Args:
        tracks: list of track dicts (from build_and_filter_tracks)
        config: full labeling_rules config dict

    Returns:
        List of merged track dicts.
    """
    merge_cfg = config.get("track_merging", {})
    if not merge_cfg.get("enabled", False) or not tracks:
        return tracks

    max_gap  = merge_cfg.get("max_gap_frames", 30)
    max_dist = merge_cfg.get("max_distance_px", 50)

    # Sort tracks by their first frame
    sorted_tracks = sorted(tracks, key=lambda t: t["entries"][0][0])
    merged_flags  = [False] * len(sorted_tracks)
    result        = []

    for i, track_i in enumerate(sorted_tracks):
        if merged_flags[i]:
            continue

        combined_entries = list(track_i["entries"])
        last_frame = track_i["entries"][-1][0]
        last_box   = track_i["entries"][-1][1]

        for j in range(i + 1, len(sorted_tracks)):
            if merged_flags[j]:
                continue
            track_j      = sorted_tracks[j]
            first_frame_j = track_j["entries"][0][0]
            first_box_j   = track_j["entries"][0][1]

            gap = first_frame_j - last_frame
            if gap > max_gap:
                break  # tracks are sorted; no later track can be closer

            # Spatial distance between endpoints
            lcx = (last_box[0] + last_box[2]) / 2
            lcy = (last_box[1] + last_box[3]) / 2
            fcx = (first_box_j[0] + first_box_j[2]) / 2
            fcy = (first_box_j[1] + first_box_j[3]) / 2
            dist = np.sqrt((lcx - fcx) ** 2 + (lcy - fcy) ** 2)

            if dist <= max_dist:
                combined_entries.extend(track_j["entries"])
                merged_flags[j] = True
                # Update last known position for chained merging
                last_frame = track_j["entries"][-1][0]
                last_box   = track_j["entries"][-1][1]

        combined_entries.sort(key=lambda e: e[0])
        result.append({
            **track_i,
            "entries":    combined_entries,
            "last_frame": combined_entries[-1][0],
            "last_box":   combined_entries[-1][1],
        })

    n_before = len(tracks)
    n_after  = len(result)
    if n_before != n_after:
        print(f"  Track merging: {n_before} → {n_after} tracks "
              f"({n_before - n_after} fragments merged)")
    else:
        print(f"  Track merging: no fragments to merge ({n_after} tracks)")
    return result


# ─────────────────────────────────────────────────────────────
# Build smooth keyhole trajectory from GDINO-detected positions
# ─────────────────────────────────────────────────────────────
def build_keyhole_trajectory(keyhole_raw, num_frames, config):
    """
    Convert sparse GDINO keyhole detections into a smooth, interpolated
    trajectory spanning from first keyhole frame to end of video.

    Steps:
      1. Temporal outlier rejection (local median filter on cx)
      2. Find first real keyhole frame (keyhole starts at right, moves left)
      3. Linear interpolation across gaps
      4. Constant extrapolation to end of video
      5. Rolling-average smoothing

    Returns:
        (keyhole_positions, first_keyhole_frame)
        keyhole_positions: {frame_idx: {"cx", "cy", "bbox"}}
        first_keyhole_frame: int or None
    """
    if not keyhole_raw:
        print("  WARNING: No keyhole detections found by GDINO.")
        return {}, None

    kh_traj = config.get("keyhole_trajectory", {})
    max_jump         = kh_traj.get("max_jump_px", 120)
    median_window    = kh_traj.get("median_window", 30)
    smooth_window    = kh_traj.get("smoothing_window", 5)
    first_frame_min_x = kh_traj.get("first_frame_min_x", 100)
    extrapolation_mode = kh_traj.get("extrapolation_mode", "linear")
    min_filtered_frames = int(kh_traj.get("min_filtered_frames", 4))

    print(f"  Raw keyhole candidates: {len(keyhole_raw)} frames")

    # Temporal filter — reject cx outliers
    filtered = _temporal_filter(keyhole_raw, max_jump, median_window)
    if len(filtered) < min_filtered_frames:
        print("  NOTE: Temporal filter too aggressive for this trajectory; "
              "falling back to raw keyhole detections.")
        filtered = dict(keyhole_raw)
    print(f"  After temporal filter: {len(filtered)} frames")

    # Find first real keyhole frame
    first_kh = _find_first_keyhole_frame(filtered, first_frame_min_x)
    if first_kh is None:
        print("  WARNING: No reliable keyhole frame found after temporal filtering.")
        return {}, None

    # Drop frames before first keyhole
    filtered = {f: c for f, c in filtered.items() if f >= first_kh}
    last_kh = max(filtered.keys())
    print(f"  Keyhole range: frames {first_kh} – {last_kh}")

    # Interpolate gaps + extrapolate to end of video
    positions = _interpolate_positions(
        filtered, first_kh, num_frames - 1, extrapolation_mode=extrapolation_mode,
    )
    print(f"  After interpolation: {len(positions)} frames")

    # Smooth
    positions = _smooth_positions(positions, smooth_window)

    return dict(positions), first_kh


def select_keyhole_track(all_detections, config, img_h, img_w):
    """
    Select a keyhole track using temporal consistency instead of shape alone.

    Builds relaxed candidate tracks, stitches fragmented tracks, then picks
    the most plausible right-to-left track using motion, leftmost-ness,
    top proximity, and constant-velocity consistency.

    Returns:
        (keyhole_raw, metrics)
    """
    kh_cfg = config.get("detection", {}).get("keyhole", {})
    track_cfg = config.get("tracking", {})
    sel_cfg = config.get("keyhole_selection", {})

    kh_threshold = float(kh_cfg.get("box_threshold", 0.0))
    max_top_y = kh_cfg.get("max_top_y", None)

    candidate_threshold = float(sel_cfg.get("track_box_threshold", max(0.01, kh_threshold * 0.5)))
    candidate_max_area_ratio = float(
        sel_cfg.get("track_max_box_area_ratio", kh_cfg.get("max_box_area_ratio", 1.0) * 1.5)
    )
    top_y_slack = float(sel_cfg.get("track_top_y_slack_px", 140.0))
    candidate_max_top_y = None if max_top_y is None else float(max_top_y + top_y_slack)
    candidate_min_height = float(sel_cfg.get("track_min_height_px", 8.0))
    base_max_width = kh_cfg.get("max_width", img_w)
    if base_max_width == float("inf"):
        base_max_width = img_w
    candidate_max_width = float(sel_cfg.get("track_max_width_px", max(base_max_width * 2.0, 120.0)))

    candidates = {}
    for fidx, (boxes, names, scores) in all_detections.items():
        c_boxes, c_names, c_scores = [], [], []
        for box, name, score in zip(boxes, names, scores):
            if score < candidate_threshold:
                continue
            x1, y1, x2, y2 = box
            w_px = x2 - x1
            h_px = y2 - y1
            if h_px < candidate_min_height or w_px > candidate_max_width:
                continue
            area_ratio = (w_px * h_px) / max(1.0, (img_h * img_w))
            if area_ratio > candidate_max_area_ratio:
                continue
            if candidate_max_top_y is not None and y1 > candidate_max_top_y:
                continue
            c_boxes.append(box)
            c_names.append(name)
            c_scores.append(score)
        if c_boxes:
            candidates[fidx] = (c_boxes, c_names, c_scores)

    if not candidates:
        return {}, {}

    frames_list = sorted(candidates.keys())
    min_track_length = int(sel_cfg.get("track_min_length", 1))
    track_iou = float(sel_cfg.get("track_iou_threshold", 0.08))
    max_gap = int(sel_cfg.get("track_max_gap", max(int(track_cfg.get("max_track_gap", 5)), 8)))

    tracks, _ = build_and_filter_tracks(
        candidates, frames_list,
        track_iou_threshold=track_iou,
        max_track_gap=max_gap,
        min_track_length=min_track_length,
    )
    if not tracks:
        return {}, {}

    def _stitch_tracks(raw_tracks):
        merge_gap = int(sel_cfg.get("merge_max_gap_frames", 20))
        merge_dist = float(sel_cfg.get("merge_max_distance_px", 120.0))
        merge_right_tol = float(sel_cfg.get("merge_right_tolerance_px", 20.0))

        sorted_tracks = sorted(raw_tracks, key=lambda t: t["entries"][0][0])
        used = [False] * len(sorted_tracks)
        stitched = []

        for i, trk in enumerate(sorted_tracks):
            if used[i]:
                continue
            used[i] = True
            combined_entries = list(trk["entries"])

            while True:
                last_frame = combined_entries[-1][0]
                last_box = combined_entries[-1][1]
                last_cx = (last_box[0] + last_box[2]) / 2.0
                last_cy = (last_box[1] + last_box[3]) / 2.0

                best_j = None
                best_cost = None
                for j, cand in enumerate(sorted_tracks):
                    if used[j]:
                        continue
                    first_entry = cand["entries"][0]
                    first_frame = first_entry[0]
                    if first_frame <= last_frame:
                        continue
                    gap = first_frame - last_frame - 1
                    if gap > merge_gap:
                        continue

                    fb = first_entry[1]
                    first_cx = (fb[0] + fb[2]) / 2.0
                    first_cy = (fb[1] + fb[3]) / 2.0
                    if first_cx > last_cx + merge_right_tol:
                        continue
                    dist = float(np.hypot(first_cx - last_cx, first_cy - last_cy))
                    if dist > merge_dist:
                        continue

                    cost = dist + 2.0 * gap
                    if best_j is None or cost < best_cost:
                        best_j = j
                        best_cost = cost

                if best_j is None:
                    break

                used[best_j] = True
                combined_entries.extend(sorted_tracks[best_j]["entries"])
                combined_entries.sort(key=lambda e: e[0])

            stitched.append({
                "id": trk.get("id", i),
                "entries": combined_entries,
                "last_frame": combined_entries[-1][0],
                "last_box": combined_entries[-1][1],
            })

        return stitched

    tracks = _stitch_tracks(tracks)
    max_frame_idx = max(all_detections.keys()) if all_detections else 0

    def _track_metrics(track):
        entries = track["entries"]
        length = len(entries)
        frames = [e[0] for e in entries]
        cxs = [(e[1][0] + e[1][2]) / 2.0 for e in entries]
        det_scores = [float(e[3]) for e in entries]
        y1s = [e[1][1] for e in entries]
        first_frame = frames[0]
        last_frame = frames[-1]
        first_cx = cxs[0]
        last_cx = cxs[-1]
        leftward_dist = first_cx - last_cx
        leftward_ratio = leftward_dist / max(1.0, img_w)

        avg_y1 = sum(y1s) / max(1, length)
        if candidate_max_top_y is not None and candidate_max_top_y > 0:
            top_ratio = max(0.0, (candidate_max_top_y - avg_y1) / candidate_max_top_y)
        else:
            top_ratio = 0.0

        # How often the track is the leftmost object in its frame.
        leftmost_scores = []
        for fidx, box, _name, _score in entries:
            frame_boxes = all_detections.get(fidx, ([], [], []))[0]
            if not frame_boxes:
                continue
            x1 = box[0]
            rank = sum(1 for b in frame_boxes if b[0] < x1)
            n = len(frame_boxes)
            leftness = 1.0 if n <= 1 else 1.0 - (rank / max(1, n - 1))
            leftmost_scores.append(leftness)
        leftmost_ratio = (
            sum(leftmost_scores) / len(leftmost_scores) if leftmost_scores else 0.0
        )

        span = last_frame - first_frame + 1
        occupancy = length / max(1, span)
        start_ratio = 1.0 - (first_frame / max(1, max_frame_idx))

        # Constant-velocity fit for cx = v*t + b
        n = length
        sum_t = float(sum(frames))
        sum_x = float(sum(cxs))
        sum_tt = float(sum(t * t for t in frames))
        sum_tx = float(sum(t * x for t, x in zip(frames, cxs)))
        denom = n * sum_tt - sum_t * sum_t
        if denom != 0:
            slope = (n * sum_tx - sum_t * sum_x) / denom
            intercept = (sum_x - slope * sum_t) / n
            residuals = [x - (slope * t + intercept) for t, x in zip(frames, cxs)]
            rms = (sum(r * r for r in residuals) / n) ** 0.5
        else:
            slope = 0.0
            rms = float("inf")

        return {
            "length": length,
            "first_frame": int(first_frame),
            "last_frame": int(last_frame),
            "leftward_px": float(leftward_dist),
            "leftward_ratio": float(leftward_ratio),
            "top_ratio": float(top_ratio),
            "leftmost_ratio": float(leftmost_ratio),
            "occupancy": float(occupancy),
            "start_ratio": float(start_ratio),
            "avg_score": float(sum(det_scores) / max(1, len(det_scores))),
            "velocity_slope": float(slope),
            "velocity_rms": float(rms),
        }

    def _score_track(track):
        m = _track_metrics(track)
        max_residual = float(sel_cfg.get("velocity_max_residual_px", 8.0))
        velocity_weight = float(sel_cfg.get("velocity_weight", 2.0))
        leftmost_weight = float(sel_cfg.get("leftmost_weight", 3.0))
        early_weight = float(sel_cfg.get("early_start_weight", 1.5))
        occupancy_weight = float(sel_cfg.get("occupancy_weight", 2.0))

        if max_residual > 0 and m["velocity_rms"] != float("inf"):
            velocity_score = max(0.0, (max_residual - m["velocity_rms"]) / max_residual)
        else:
            velocity_score = 0.0

        score = (
            1.2 * m["length"]
            + 6.0 * m["leftward_ratio"]
            + 2.0 * m["top_ratio"]
            + leftmost_weight * m["leftmost_ratio"]
            + early_weight * m["start_ratio"]
            + occupancy_weight * m["occupancy"]
            + velocity_weight * velocity_score
        )

        if m["leftward_px"] < 0:
            score -= 8.0 * abs(m["leftward_ratio"])
        if m["velocity_slope"] > 0:
            score -= 4.0 * abs(m["velocity_slope"])
        return score, m

    best = None
    best_score = None
    best_metrics = None
    for t in tracks:
        score, metrics = _score_track(t)
        if best is None or score > best_score:
            best = t
            best_score = score
            best_metrics = metrics

    if best is None:
        return {}, {}

    keyhole_raw = {}
    for fidx, box, _name, score in best["entries"]:
        x1, y1, x2, y2 = box
        keyhole_raw[fidx] = {
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "score": float(score),
        }

    return keyhole_raw, (best_metrics or {})


def recover_keyhole_track_with_template(
    keyhole_raw, all_detections, frame_names, source_dir, config, crop_bottom_height=None,
):
    """
    Recover keyhole continuity when detector track stalls or drops.

    Strategy:
      1) Use previous keyhole box + velocity as prediction.
      2) Try nearby detection candidates in current frame.
      3) Fallback to template matching around predicted location.
      4) If both fail, propagate previous box with motion prior.

    This function updates keyhole trajectory only (bubble detections are not
    modified here), so bubble detection logic remains stable.
    """
    rcfg = config.get("keyhole_recovery", {})
    if not rcfg.get("enabled", False):
        return keyhole_raw
    if not keyhole_raw or not frame_names:
        return keyhole_raw
    if len(keyhole_raw) < int(rcfg.get("min_seed_frames", 3)):
        return keyhole_raw

    search_radius_x = int(rcfg.get("search_radius_x", 90))
    search_radius_y = int(rcfg.get("search_radius_y", 40))
    template_min_score = float(rcfg.get("template_match_min_score", 0.25))
    detection_min_score = float(
        rcfg.get("detection_min_score", config.get("detection", {}).get("keyhole", {}).get("box_threshold", 0.15) * 0.6)
    )
    detection_max_dist = float(rcfg.get("detection_max_dist_px", 120.0))
    max_right_drift = float(rcfg.get("max_right_drift_px", 8.0))
    max_step_px = float(rcfg.get("max_step_px", 40.0))
    min_left_step_px = float(rcfg.get("min_left_step_px", 0.8))
    size_change_tol = float(rcfg.get("size_change_tolerance", 2.5))
    max_pred_only_gap = int(rcfg.get("max_pred_only_gap", 2))
    max_recover_horizon = int(rcfg.get("max_recover_horizon_frames", 12))

    # Cache grayscale frames for template matching.
    gray_cache = {}

    def _get_gray(frame_idx):
        if frame_idx in gray_cache:
            return gray_cache[frame_idx]
        if frame_idx < 0 or frame_idx >= len(frame_names):
            return None
        fpath = os.path.join(source_dir, frame_names[frame_idx])
        img_rgb = read_image_any(fpath)
        if crop_bottom_height is not None and img_rgb.shape[0] > crop_bottom_height:
            img_rgb = img_rgb[-crop_bottom_height:, :, :]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_cache[frame_idx] = gray
        return gray

    def _clamp_box(x1, y1, x2, y2, img_w_, img_h_):
        x1 = max(0.0, min(float(img_w_ - 2), x1))
        y1 = max(0.0, min(float(img_h_ - 2), y1))
        x2 = max(x1 + 1.0, min(float(img_w_ - 1), x2))
        y2 = max(y1 + 1.0, min(float(img_h_ - 1), y2))
        return x1, y1, x2, y2

    def _entry_from_box(box, score):
        x1, y1, x2, y2 = box
        return {
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "score": float(score),
        }

    def _shift_box(entry, dx, dy):
        x1 = entry["x"] + dx
        y1 = entry["y"] + dy
        x2 = entry["x"] + entry["w"] + dx
        y2 = entry["y"] + entry["h"] + dy
        return [x1, y1, x2, y2]

    num_frames = len(frame_names)
    sorted_known = sorted(f for f in keyhole_raw.keys() if 0 <= f < num_frames)
    if not sorted_known:
        return keyhole_raw

    start_f = sorted_known[0]
    recovered = {start_f: dict(keyhole_raw[start_f])}
    pred_only_streak = 0

    # Include any known detections before start frame as-is.
    for fidx in sorted_known:
        if fidx < start_f:
            recovered[fidx] = dict(keyhole_raw[fidx])

    for fidx in range(start_f + 1, num_frames):
        if (fidx - start_f) > max_recover_horizon and fidx not in keyhole_raw:
            continue

        prev = recovered.get(fidx - 1)
        if prev is None:
            # No propagated state for this frame; keep detector result if present.
            if fidx in keyhole_raw:
                recovered[fidx] = dict(keyhole_raw[fidx])
                pred_only_streak = 0
            continue

        # Estimate velocity from last two recovered states.
        if (fidx - 2) in recovered:
            prev2 = recovered[fidx - 2]
            vx = prev["cx"] - prev2["cx"]
            vy = prev["cy"] - prev2["cy"]
        else:
            vx, vy = -min_left_step_px, 0.0

        # Keyhole should move right->left; damp rightward motion.
        if vx > max_right_drift:
            vx = max_right_drift
        if vx > 0:
            vx = max(0.0, vx - 0.5)
        if vx < -max_step_px:
            vx = -max_step_px
        if abs(vx) < min_left_step_px:
            vx = -min_left_step_px

        pred_box = _shift_box(prev, vx, vy)

        # Candidate A: detection near prediction.
        best_det = None
        best_det_cost = None
        d_boxes, _d_names, d_scores = all_detections.get(fidx, ([], [], []))
        prev_area = max(1.0, prev["w"] * prev["h"])
        pred_cx = (pred_box[0] + pred_box[2]) / 2.0
        pred_cy = (pred_box[1] + pred_box[3]) / 2.0
        for box, score in zip(d_boxes, d_scores):
            if score < detection_min_score:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = float(np.hypot(cx - pred_cx, cy - pred_cy))
            if dist > detection_max_dist:
                continue
            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            area_ratio = box_area / prev_area
            if area_ratio < (1.0 / size_change_tol) or area_ratio > size_change_tol:
                continue
            right_penalty = max(0.0, cx - (prev["cx"] + max_right_drift))
            cost = dist + 4.0 * right_penalty - 25.0 * float(score)
            if best_det is None or cost < best_det_cost:
                best_det = box
                best_det_cost = cost

        # Candidate B: template matching around prediction.
        tm_box = None
        tm_score = -1.0
        prev_gray = _get_gray(fidx - 1)
        curr_gray = _get_gray(fidx)
        if prev_gray is not None and curr_gray is not None:
            x1 = int(round(prev["x"]))
            y1 = int(round(prev["y"]))
            x2 = int(round(prev["x"] + prev["w"]))
            y2 = int(round(prev["y"] + prev["h"]))
            x1 = max(0, min(prev_gray.shape[1] - 2, x1))
            y1 = max(0, min(prev_gray.shape[0] - 2, y1))
            x2 = max(x1 + 2, min(prev_gray.shape[1] - 1, x2))
            y2 = max(y1 + 2, min(prev_gray.shape[0] - 1, y2))
            template = prev_gray[y1:y2, x1:x2]

            if template.size > 0:
                pw = x2 - x1
                ph = y2 - y1
                pcx = int(round(pred_cx))
                pcy = int(round(pred_cy))
                sx1 = max(0, pcx - search_radius_x - pw // 2)
                sy1 = max(0, pcy - search_radius_y - ph // 2)
                sx2 = min(curr_gray.shape[1], pcx + search_radius_x + pw // 2)
                sy2 = min(curr_gray.shape[0], pcy + search_radius_y + ph // 2)
                search = curr_gray[sy1:sy2, sx1:sx2]

                if search.shape[0] >= ph and search.shape[1] >= pw:
                    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
                    mx, my = max_l
                    tx1 = float(sx1 + mx)
                    ty1 = float(sy1 + my)
                    tx2 = tx1 + pw
                    ty2 = ty1 + ph
                    tm_box = [tx1, ty1, tx2, ty2]
                    tm_score = float(max_v)

        # Pick best source.
        chosen_box = None
        chosen_score = prev.get("score", 0.0)
        if best_det is not None:
            chosen_box = list(best_det)
            # Use detector score if available for this exact box.
            chosen_score = max(chosen_score, detection_min_score)
            pred_only_streak = 0
        elif tm_box is not None and tm_score >= template_min_score:
            chosen_box = tm_box
            chosen_score = max(chosen_score, tm_score)
            pred_only_streak = 0

        # Final fallback: propagate prediction.
        if chosen_box is None:
            pred_only_streak += 1
            if pred_only_streak > max_pred_only_gap:
                # Stop free-running prediction to avoid drift.
                continue
            chosen_box = pred_box

        # Enforce monotonic direction softly.
        c_cx = (chosen_box[0] + chosen_box[2]) / 2.0
        if c_cx > prev["cx"] + max_right_drift:
            drift = c_cx - (prev["cx"] + max_right_drift)
            chosen_box[0] -= drift
            chosen_box[2] -= drift

        gray = _get_gray(fidx)
        if gray is None:
            continue
        x1, y1, x2, y2 = _clamp_box(chosen_box[0], chosen_box[1], chosen_box[2], chosen_box[3], gray.shape[1], gray.shape[0])
        recovered[fidx] = _entry_from_box([x1, y1, x2, y2], chosen_score)

        # If we also had a detector keyhole on this frame, blend a little for stability.
        if fidx in keyhole_raw:
            det = keyhole_raw[fidx]
            recovered[fidx]["cx"] = 0.7 * recovered[fidx]["cx"] + 0.3 * det["cx"]
            recovered[fidx]["cy"] = 0.7 * recovered[fidx]["cy"] + 0.3 * det["cy"]

    # Keep original known frames if recovery skipped any.
    for fidx, v in keyhole_raw.items():
        recovered.setdefault(fidx, dict(v))

    return recovered


def score_keyhole_raw(keyhole_raw):
    if not keyhole_raw:
        return {
            "length": 0,
            "leftward_px": 0.0,
            "first_cx": None,
            "last_cx": None,
        }
    frames = sorted(keyhole_raw.keys())
    first_cx = keyhole_raw[frames[0]]["cx"]
    last_cx = keyhole_raw[frames[-1]]["cx"]
    leftward_px = first_cx - last_cx
    return {
        "length": len(frames),
        "leftward_px": float(leftward_px),
        "first_cx": float(first_cx),
        "last_cx": float(last_cx),
    }


def filter_bubbles_by_keyhole_side(
    bubble_detections, keyhole_positions, config,
    keyhole_observed_frames=None, frame_width=None,
):
    """
    Remove bubble detections that are left of the keyhole in the same frame.

    Returns:
        (filtered_bubble_detections, num_removed)
    """
    validity_cfg = config.get("bubble_validity", {})
    if not validity_cfg.get("require_right_of_keyhole", True):
        return bubble_detections, 0

    min_dx = float(validity_cfg.get("min_dx_from_keyhole_px", 0.0))
    apply_only_when_observed = bool(validity_cfg.get("apply_only_when_observed", True))
    observed_tol = int(validity_cfg.get("observed_frame_tolerance", 2))
    ignore_if_keyhole_out_of_frame = bool(validity_cfg.get("ignore_if_keyhole_out_of_frame", True))

    reliable_frames = None
    if apply_only_when_observed and keyhole_observed_frames is not None:
        reliable_frames = set()
        for fidx in keyhole_observed_frames:
            for ff in range(fidx - observed_tol, fidx + observed_tol + 1):
                if ff >= 0:
                    reliable_frames.add(ff)

    filtered = {}
    removed = 0

    for fidx, (boxes, names, scores) in bubble_detections.items():
        if fidx not in keyhole_positions:
            filtered[fidx] = (boxes, names, scores)
            continue
        if reliable_frames is not None and fidx not in reliable_frames:
            filtered[fidx] = (boxes, names, scores)
            continue

        kh_cx = keyhole_positions[fidx]["cx"]
        if ignore_if_keyhole_out_of_frame and frame_width is not None:
            if kh_cx < 0 or kh_cx > frame_width:
                filtered[fidx] = (boxes, names, scores)
                continue

        keep_boxes, keep_names, keep_scores = [], [], []
        for box, name, score in zip(boxes, names, scores):
            bubble_cx = (box[0] + box[2]) / 2.0
            if bubble_cx >= (kh_cx + min_dx):
                keep_boxes.append(box)
                keep_names.append(name)
                keep_scores.append(score)
            else:
                removed += 1

        filtered[fidx] = (keep_boxes, keep_names, keep_scores)

    return filtered, removed


def clamp_keyhole_positions_to_frame(keyhole_positions, img_w, img_h):
    """
    Clamp keyhole bbox/cx/cy to image bounds to avoid off-frame drift artifacts.
    """
    if not keyhole_positions:
        return keyhole_positions

    clamped = {}
    for fidx, kh in keyhole_positions.items():
        bbox = kh.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            clamped[fidx] = kh
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(img_w - 2), x1))
        y1 = max(0.0, min(float(img_h - 2), y1))
        x2 = max(x1 + 1.0, min(float(img_w - 1), x2))
        y2 = max(y1 + 1.0, min(float(img_h - 1), y2))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        clamped[fidx] = {"cx": cx, "cy": cy, "bbox": [x1, y1, x2, y2]}

    return clamped


def normalize_keyhole_track_output(track_output):
    """
    Defensive normalizer for select_keyhole_track() outputs.

    Always returns:
        (keyhole_raw_dict, metrics_dict, valid_return_shape)
    """
    if isinstance(track_output, tuple):
        if len(track_output) >= 2:
            kraw = track_output[0] if isinstance(track_output[0], dict) else {}
            metrics = track_output[1] if isinstance(track_output[1], dict) else {}
            return kraw, metrics, True
        if len(track_output) == 1:
            kraw = track_output[0] if isinstance(track_output[0], dict) else {}
            return kraw, {}, False
        return {}, {}, False

    if isinstance(track_output, dict):
        return track_output, {}, False

    return {}, {}, False


def find_keyhole_stop_frame(keyhole_positions, first_keyhole_frame, config):
    """
    Detect when the keyhole stops moving right-to-left (cx no longer decreases).

    Returns:
        last_valid_frame (int) or None if no stop detected.
    """
    if not keyhole_positions or first_keyhole_frame is None:
        return None

    kh_traj = config.get("keyhole_trajectory", {})
    min_leftward = kh_traj.get("stop_min_leftward_px", 0.5)
    still_window = int(kh_traj.get("stop_still_window", 5))
    left_edge_x = kh_traj.get("stop_left_edge_x", None)
    if still_window <= 0:
        return None

    frames = sorted(f for f in keyhole_positions.keys() if f >= first_keyhole_frame)
    if len(frames) < 2:
        return None

    still_count = 0
    for i in range(1, len(frames)):
        prev_cx = keyhole_positions[frames[i - 1]]["cx"]
        curr_cx = keyhole_positions[frames[i]]["cx"]
        leftward = prev_cx - curr_cx
        in_left_zone = True
        if left_edge_x is not None:
            in_left_zone = max(prev_cx, curr_cx) <= left_edge_x

        if not in_left_zone:
            still_count = 0
            continue
        if leftward < min_leftward:
            still_count += 1
            if still_count >= still_window:
                stop_start = frames[i - still_window + 1]
                last_valid = max(first_keyhole_frame, stop_start - 1)
                return last_valid
        else:
            still_count = 0

    return None


def estimate_end_frame_from_keyhole_loss(observed_keyhole_frames, num_frames, config):
    """
    Estimate trajectory end when keyhole detections are lost.

    Returns:
        frame index or None
    """
    kh_traj = config.get("keyhole_trajectory", {})
    if not kh_traj.get("end_on_keyhole_loss", True):
        return None
    if not observed_keyhole_frames:
        return None

    min_observed = int(kh_traj.get("min_observed_frames_for_loss_rule", 3))
    if len(observed_keyhole_frames) < min_observed:
        return None

    last_obs = max(observed_keyhole_frames)
    if last_obs >= num_frames - 1:
        return None

    tail = int(kh_traj.get("max_unobserved_tail_frames", 8))
    tail = max(0, tail)
    return min(num_frames - 1, last_obs + tail)


def validate_and_adjust_end_with_post_end_evidence(
    candidate_end_frame, all_detections, num_frames, config, img_h, img_w,
):
    """
    Validate trajectory-end candidate using post-end keyhole-like evidence.

    If a sufficiently long, temporally consistent keyhole-evidence run exists
    after candidate_end_frame, extend the end frame to avoid early truncation.
    """
    val_cfg = config.get("keyhole_end_validation", {})
    if not val_cfg.get("enabled", True):
        return candidate_end_frame, None
    if candidate_end_frame is None or candidate_end_frame >= num_frames - 1:
        return candidate_end_frame, None

    kh_cfg = config.get("detection", {}).get("keyhole", {})
    kh_traj = config.get("keyhole_trajectory", {})

    box_threshold = float(val_cfg.get("box_threshold", kh_cfg.get("box_threshold", 0.15) * 0.55))
    min_run_frames = int(val_cfg.get("min_run_frames", 5))
    max_gap_frames = int(val_cfg.get("max_gap_frames", 2))
    min_avg_score = float(val_cfg.get("min_avg_score", 0.08))
    max_right_slope = float(val_cfg.get("max_rightward_slope_px_per_frame", 0.3))
    left_edge_margin = float(val_cfg.get("left_edge_margin_px", 8.0))
    extend_tail = int(val_cfg.get("extend_tail_frames", 6))

    min_h = float(kh_cfg.get("min_height", 0))
    max_w = kh_cfg.get("max_width", float("inf"))
    min_ar = float(kh_cfg.get("min_aspect_ratio", 0.0))
    max_top_y = kh_cfg.get("max_top_y", float("inf"))
    max_area = float(kh_cfg.get("max_box_area_ratio", 1.0))

    min_h_scale = float(val_cfg.get("min_height_scale", 0.4))
    max_w_scale = float(val_cfg.get("max_width_scale", 2.5))
    min_ar_scale = float(val_cfg.get("min_aspect_ratio_scale", 0.3))
    max_area_scale = float(val_cfg.get("max_area_ratio_scale", 2.0))
    top_y_slack = float(val_cfg.get("top_y_slack_px", 140.0))

    left_edge_x = float(kh_traj.get("stop_left_edge_x", 70))
    frame_area = max(1.0, img_h * img_w)

    # Build frame -> best (leftmost) keyhole-like candidate after candidate_end_frame.
    evidence = {}
    for fidx in range(candidate_end_frame + 1, num_frames):
        boxes, _names, scores = all_detections.get(fidx, ([], [], []))
        best = None
        best_key = None
        for box, score in zip(boxes, scores):
            if score < box_threshold:
                continue
            x1, y1, x2, y2 = box
            h_px = y2 - y1
            w_px = x2 - x1
            if h_px < (min_h * min_h_scale):
                continue
            if max_w != float("inf") and w_px > (max_w * max_w_scale):
                continue
            ar = h_px / w_px if w_px > 0 else 0.0
            if ar < (min_ar * min_ar_scale):
                continue
            if y1 > (max_top_y + top_y_slack):
                continue
            area_ratio = (h_px * w_px) / frame_area
            if area_ratio > (max_area * max_area_scale):
                continue

            # Prefer leftmost, then higher score.
            key = (x1, -float(score))
            if best is None or key < best_key:
                best_key = key
                best = {
                    "cx": (x1 + x2) / 2.0,
                    "score": float(score),
                }

        if best is not None:
            evidence[fidx] = best

    if not evidence:
        return candidate_end_frame, None

    # Group into near-contiguous runs with small allowed gaps.
    frames = sorted(evidence.keys())
    runs = []
    current = [frames[0]]
    for fidx in frames[1:]:
        if (fidx - current[-1]) <= (max_gap_frames + 1):
            current.append(fidx)
        else:
            runs.append(current)
            current = [fidx]
    runs.append(current)

    qualified = []
    for run in runs:
        if len(run) < min_run_frames:
            continue
        t = np.array(run, dtype=np.float32)
        x = np.array([evidence[f]["cx"] for f in run], dtype=np.float32)
        s = np.array([evidence[f]["score"] for f in run], dtype=np.float32)

        t_center = t - t.mean()
        denom = float(np.sum(t_center ** 2))
        slope = float(np.sum(t_center * (x - x.mean())) / denom) if denom > 0 else 0.0
        avg_score = float(np.mean(s))
        median_cx = float(np.median(x))

        if avg_score < min_avg_score:
            continue
        if median_cx <= (left_edge_x + left_edge_margin):
            continue
        if slope > max_right_slope:
            continue

        qualified.append({
            "start": run[0],
            "end": run[-1],
            "len": len(run),
            "avg_score": avg_score,
            "median_cx": median_cx,
            "slope": slope,
        })

    if not qualified:
        return candidate_end_frame, None

    latest_end = max(q["end"] for q in qualified)
    adjusted_end = min(num_frames - 1, latest_end + max(0, extend_tail))
    if adjusted_end <= candidate_end_frame:
        return candidate_end_frame, None

    best_q = max(qualified, key=lambda q: q["end"])
    info = {
        "old_end": candidate_end_frame,
        "new_end": adjusted_end,
        "run_start": best_q["start"],
        "run_end": best_q["end"],
        "run_len": best_q["len"],
        "avg_score": best_q["avg_score"],
        "median_cx": best_q["median_cx"],
        "slope": best_q["slope"],
    }
    return adjusted_end, info


# ─────────────────────────────────────────────────────────────
# Classify bubble tracks by proximity & persistence
# ─────────────────────────────────────────────────────────────
def classify_bubble_tracks(bubble_tracks, keyhole_positions, config):
    """
    For each bubble track, compute:
      - average distance to keyhole center (metadata only)
      - near/far classification (metadata only)
      - transient/permanent classification (used for labels 3/4)

    Returns:
        List of track dicts with added classification metadata.
    """
    prox_cfg  = config["proximity"]
    track_cfg = config["track_classification"]

    near_threshold = prox_cfg["near_threshold_pixels"]
    transient_max  = track_cfg["transient_max_frames"]
    permanent_min  = track_cfg["permanent_min_frames"]
    method         = prox_cfg.get("method", "center_distance")

    classified = []
    for track in bubble_tracks:
        entries  = track["entries"]
        duration = len(entries)

        distances = []
        for entry in entries:
            fidx, box = entry[0], entry[1]
            bubble_cx = (box[0] + box[2]) / 2
            bubble_cy = (box[1] + box[3]) / 2

            if fidx in keyhole_positions:
                kh     = keyhole_positions[fidx]
                kh_cx  = kh["cx"]
                kh_cy  = kh["cy"]
                kh_box = kh["bbox"]
                if method == "edge_distance":
                    dx   = max(0, max(box[0] - kh_box[2], kh_box[0] - box[2]))
                    dy   = max(0, max(box[1] - kh_box[3], kh_box[1] - box[3]))
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    dist = np.sqrt(
                        (bubble_cx - kh_cx) ** 2 + (bubble_cy - kh_cy) ** 2
                    )
                distances.append(dist)

        avg_distance = float(np.mean(distances)) if distances else float("inf")
        is_near      = avg_distance <= near_threshold
        is_permanent = duration >= permanent_min
        is_transient = duration <  transient_max

        classified.append({
            **track,
            "duration":                duration,
            "avg_distance_to_keyhole": round(avg_distance, 2),
            "is_near_keyhole":         is_near,
            "is_permanent":            is_permanent,
            "is_transient":            is_transient,
        })

    near_count = sum(1 for t in classified if t["is_near_keyhole"])
    perm_count = sum(1 for t in classified if t["is_permanent"])
    print(
        f"  Bubble tracks: {len(classified)} total, "
        f"{near_count} near keyhole, {perm_count} permanent"
    )
    return classified


# ─────────────────────────────────────────────────────────────
# Per-frame labeling — corrected rules
# ─────────────────────────────────────────────────────────────
def label_frames(
    num_frames, keyhole_positions,
    classified_bubble_tracks,
):
    """
    Assign a label to each frame:

      0 - No Signal:
            No keyhole detected at this frame.

      1 - Normal Process:
            Keyhole present, AND across ALL frames in the entire video
            there are NO bubbles at all.

      2 - Unstable Process without Pore Generation:
            Keyhole present, no bubble at THIS frame, but bubbles DO
            exist somewhere across all frames in the video.

      3 - Transient Pore Generation:
            At least one bubble at this frame will eventually disappear.
            TRANSIENT TAKES PRIORITY: even if permanent bubbles also exist
            at this frame, the presence of ANY transient bubble makes the
            frame label 3.

      4 - Permanent Pore Generation:
            ALL bubbles at this frame are permanent (none are transient).

    Priority hierarchy (highest wins):
        0  (no keyhole)  is determined first,
        1  (no bubbles globally) next,
        2  (no bubbles here, but exist elsewhere) next,
        3  (any transient bubble here) beats 4,
        4  (only permanent bubbles here).
    """
    # Build per-frame bubble track lookup
    frame_tracks = defaultdict(list)
    for track in classified_bubble_tracks:
        for entry in track["entries"]:
            frame_tracks[entry[0]].append(track)

    # Global: does ANY bubble track exist in the entire video?
    has_any_bubbles = len(classified_bubble_tracks) > 0

    frame_labels = {}
    for fidx in range(num_frames):
        has_keyhole     = fidx in keyhole_positions
        tracks_here     = frame_tracks.get(fidx, [])
        has_bubbles_now = len(tracks_here) > 0

        # ── Label 0: No keyhole ──────────────────────────────────
        if not has_keyhole:
            frame_labels[fidx] = 0
            continue

        # ── Label 1: Keyhole + no bubbles anywhere in video ──────
        if not has_any_bubbles:
            frame_labels[fidx] = 1
            continue

        # ── Label 2: Keyhole + no bubbles at this frame ──────────
        if not has_bubbles_now:
            frame_labels[fidx] = 2
            continue

        # ── Labels 3 / 4: per-bubble classification ──────────────
        # Transient takes priority: if ANY bubble at this frame is
        # transient (will disappear), label the frame as 3.
        # Only label 4 when every bubble here is permanent.
        has_transient_here = any(t["is_transient"] for t in tracks_here)
        frame_labels[fidx] = 3 if has_transient_here else 4

    return frame_labels


# ─────────────────────────────────────────────────────────────
# Label smoothing (majority vote)
# ─────────────────────────────────────────────────────────────
def smooth_labels(frame_labels, config):
    """Apply majority-vote smoothing within a sliding window."""
    window = config["intervals"].get("smoothing_window", 1)
    if window <= 1:
        return frame_labels

    sorted_frames  = sorted(frame_labels.keys())
    labels_array   = [frame_labels[f] for f in sorted_frames]
    smoothed       = {}

    for i, fidx in enumerate(sorted_frames):
        start = max(0, i - window // 2)
        end   = min(len(sorted_frames), i + window // 2 + 1)
        window_labels = labels_array[start:end]
        counts        = Counter(window_labels)
        most_common   = counts.most_common()
        top_count     = most_common[0][1]
        candidates    = [lbl for lbl, cnt in most_common if cnt == top_count]
        if labels_array[i] in candidates:
            smoothed[fidx] = labels_array[i]
        else:
            smoothed[fidx] = most_common[0][0]

    return smoothed


# ─────────────────────────────────────────────────────────────
# Group into time intervals
# ─────────────────────────────────────────────────────────────
def group_into_intervals(frame_labels, num_frames, config):
    """
    Group consecutive frames with the same label into intervals.

    Returns:
        List of {start_frame, end_frame, duration_frames, label_id, label_name}.
    """
    labels_cfg    = config["labels"]
    label_name_map = {v["id"]: v["name"] for v in labels_cfg.values()}

    intervals      = []
    current_label  = None
    current_start  = None

    for fidx in range(num_frames):
        label = frame_labels.get(fidx, 0)
        if label != current_label:
            if current_label is not None:
                intervals.append({
                    "start_frame":    current_start,
                    "end_frame":      fidx - 1,
                    "duration_frames": fidx - current_start,
                    "label_id":       current_label,
                    "label_name":     label_name_map.get(current_label, "Unknown"),
                })
            current_label = label
            current_start = fidx

    if current_label is not None:
        intervals.append({
            "start_frame":    current_start,
            "end_frame":      num_frames - 1,
            "duration_frames": num_frames - current_start,
            "label_id":       current_label,
            "label_name":     label_name_map.get(current_label, "Unknown"),
        })

    min_len = config["intervals"].get("min_interval_length", 1)
    if min_len > 1:
        intervals = _merge_short_intervals(intervals, min_len)

    return intervals


def _merge_short_intervals(intervals, min_length):
    """Merge intervals shorter than min_length into their longest neighbor."""
    if len(intervals) <= 1:
        return intervals

    merged  = list(intervals)
    changed = True
    while changed:
        changed    = False
        new_merged = []
        i          = 0
        while i < len(merged):
            iv = merged[i]
            if iv["duration_frames"] < min_length and len(new_merged) > 0:
                new_merged[-1]["end_frame"]      = iv["end_frame"]
                new_merged[-1]["duration_frames"] = (
                    new_merged[-1]["end_frame"] - new_merged[-1]["start_frame"] + 1
                )
                changed = True
            elif iv["duration_frames"] < min_length and i + 1 < len(merged):
                merged[i + 1]["start_frame"]     = iv["start_frame"]
                merged[i + 1]["duration_frames"] = (
                    merged[i + 1]["end_frame"] - iv["start_frame"] + 1
                )
                changed = True
            else:
                new_merged.append(iv)
            i += 1
        merged = new_merged

    return merged


# ─────────────────────────────────────────────────────────────
# Generate labeled frames
# ─────────────────────────────────────────────────────────────
def generate_labeled_frames(
    frame_names, source_dir, output_dir, frame_labels,
    keyhole_positions, classified_bubble_tracks, config,
    crop_bottom_height=None,
):
    """
    Save annotated frames with a colored label bar, keyhole box (white),
    and per-track bubble boxes (color-coded by permanence).
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_cfg     = config["labels"]
    label_name_map  = {v["id"]: v["name"]          for v in labels_cfg.values()}
    label_color_map = {v["id"]: tuple(v["color"])   for v in labels_cfg.values()}

    # Build per-frame bubble box lookup
    frame_bubble_boxes = defaultdict(list)
    for track in classified_bubble_tracks:
        for entry in track["entries"]:
            fidx, box = entry[0], entry[1]
            frame_bubble_boxes[fidx].append((box, track))

    for fidx in tqdm(range(len(frame_names)), desc="Saving labeled frames"):
        fpath = os.path.join(source_dir, frame_names[fidx])
        img_rgb = read_image_any(fpath)
        if crop_bottom_height is not None and img_rgb.shape[0] > crop_bottom_height:
            img_rgb = img_rgb[-crop_bottom_height:, :, :]
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if img is None:
            continue

        label_id   = frame_labels.get(fidx, 0)
        label_name = label_name_map.get(label_id, "Unknown")
        color_rgb  = label_color_map.get(label_id, (128, 128, 128))
        color_bgr  = (color_rgb[2], color_rgb[1], color_rgb[0])

        w = img.shape[1]

        # Colored label bar at top
        bar_h = 30
        cv2.rectangle(img, (0, 0), (w, bar_h), color_bgr, -1)
        cv2.putText(
            img, f"[{label_id}] {label_name}  (frame {fidx})",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

        # Keyhole box (white)
        if fidx in keyhole_positions:
            kh     = keyhole_positions[fidx]
            kh_box = kh["bbox"]
            pt1    = (int(kh_box[0]), int(kh_box[1]))
            pt2    = (int(kh_box[2]), int(kh_box[3]))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
            cv2.putText(
                img, "keyhole", (pt1[0], max(pt1[1] - 4, bar_h + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

        # Bubble boxes — red=permanent, orange=transient, cyan=far-from-keyhole
        for box, track in frame_bubble_boxes.get(fidx, []):
            if track["is_permanent"]:
                bb_color = (0, 0, 255)       # red
            elif track["is_near_keyhole"]:
                bb_color = (0, 165, 255)     # orange
            else:
                bb_color = (0, 255, 255)     # cyan
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, bb_color, 1)
            cv2.putText(
                img, f"T{track['id']}", (pt1[0], max(pt1[1] - 4, bar_h + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, bb_color, 1,
            )

        cv2.imwrite(os.path.join(output_dir, f"{fidx:05d}.jpg"), img)

    print(f"Labeled frames saved to {output_dir}")


# ─────────────────────────────────────────────────────────────
# Detection cache helpers
# ─────────────────────────────────────────────────────────────
def save_detection_cache(all_detections, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialisable = {
        str(k): (v[0], v[1], v[2]) for k, v in all_detections.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f)
    print(f"Detection cache saved to {path}")


def load_detection_cache(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): (v[0], v[1], v[2]) for k, v in raw.items()}


def list_frame_files(source_dir):
    frame_names = [
        p for p in os.listdir(source_dir)
        if os.path.splitext(p)[-1].lower() in FRAME_EXTS
    ]

    def _sort_key(name):
        stem = os.path.splitext(name)[0]
        try:
            return (0, int(stem))
        except ValueError:
            nums = re.findall(r"\d+", stem)
            if nums:
                return (1, int(nums[-1]), stem)
            return (2, stem)

    frame_names.sort(key=_sort_key)
    return frame_names


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Label video frames based on keyhole and bubble detection"
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
        "--crop-bottom-height", type=int, default=CROP_BOTTOM_HEIGHT,
        help=f"Crop to bottom N pixels (default: {CROP_BOTTOM_HEIGHT})",
    )
    args = parser.parse_args()

    video_path = args.video_path
    frames_dir = args.frames_dir
    detection_cache_path = args.detection_cache
    output_frames_dir = args.output_frames_dir
    crop_bottom_height = args.crop_bottom_height

    # ── Load config ──────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")

    # ── Load models (not needed if skipping detection) ──────────
    if not args.skip_detection:
        gdino_processor, gdino_model, image_predictor, device = load_models(
            GDINO_MODEL_ID, SAM2_CHECKPOINT, SAM2_MODEL_CFG,
        )
    else:
        gdino_processor = gdino_model = image_predictor = device = None

    # ── Extract / reuse frames ───────────────────────────────────
    if args.skip_extraction:
        frame_names = list_frame_files(frames_dir)
        if not frame_names:
            raise FileNotFoundError(
                f"No frames found in {frames_dir}. Expected files with extensions: {FRAME_EXTS}"
            )
        sample = read_image_any(os.path.join(frames_dir, frame_names[0]))
        if crop_bottom_height is not None and sample.shape[0] > crop_bottom_height:
            sample = sample[-crop_bottom_height:, :, :]
        img_h, img_w = sample.shape[:2]
        print(f"Reusing {len(frame_names)} frames from {frames_dir}")
    else:
        frame_names, img_w, img_h = extract_and_crop_frames(
            video_path, frames_dir, crop_top=CROP_TOP,
        )

    num_frames = len(frame_names)
    print(f"Total frames: {num_frames}, image size: {img_w}x{img_h}")

    # ── GDINO detection ──────────────────────────────────────────
    print("\n--- Running GDINO detection ---")
    if args.skip_detection:
        print(f"  Loading from cache: {detection_cache_path}")
        all_detections = load_detection_cache(detection_cache_path)
    else:
        all_detections = run_all_detections(
            frame_names, frames_dir, config,
            gdino_processor, gdino_model, img_h, img_w, device,
            crop_bottom_height=crop_bottom_height,
        )
        save_detection_cache(all_detections, detection_cache_path)

    # ── Split detections: keyhole vs bubbles ─────────────────────
    print("\n--- Splitting detections by shape ---")
    keyhole_raw_shape, bubble_detections_shape = split_keyhole_and_bubbles(
        all_detections, config, img_h, img_w,
    )

    track_output = select_keyhole_track(all_detections, config, img_h, img_w)
    keyhole_raw_track, track_metrics, valid_track_return = normalize_keyhole_track_output(track_output)
    if not valid_track_return:
        print("  WARNING: select_keyhole_track returned unexpected shape; falling back safely.")

    sel_cfg = config.get("keyhole_selection", {})
    method = sel_cfg.get("method", "hybrid").lower()
    track_min_frames = int(sel_cfg.get("track_min_frames", 5))
    track_prefer_ratio = float(sel_cfg.get("track_prefer_ratio", 0.6))
    track_min_leftward = float(sel_cfg.get("track_min_leftward_px", 5.0))
    track_min_leftmost = float(sel_cfg.get("track_min_leftmost_ratio", 0.45))
    velocity_require = bool(sel_cfg.get("velocity_require", True))
    velocity_max_residual = float(sel_cfg.get("velocity_max_residual_px", 8.0))
    rescue_when_shape_missing = bool(sel_cfg.get("rescue_when_shape_missing", True))
    rescue_min_track_frames = int(sel_cfg.get("rescue_min_track_frames", 2))
    rescue_min_avg_score = float(sel_cfg.get("rescue_min_avg_score", 0.08))

    shape_score = score_keyhole_raw(keyhole_raw_shape)
    track_score = score_keyhole_raw(keyhole_raw_track)

    use_track = False
    velocity_ok = True
    leftmost_ok = True
    if velocity_require and track_metrics:
        if track_metrics.get("velocity_rms", float("inf")) > velocity_max_residual:
            velocity_ok = False
    if track_metrics:
        if track_metrics.get("leftmost_ratio", 0.0) < track_min_leftmost:
            leftmost_ok = False
    if method == "track":
        use_track = track_score["length"] >= track_min_frames and velocity_ok and leftmost_ok
    elif method == "shape":
        use_track = False
    else:
        if (track_score["length"] >= track_min_frames and track_score["leftward_px"] >= track_min_leftward
                and velocity_ok and leftmost_ok):
            if shape_score["length"] == 0:
                use_track = True
            else:
                use_track = track_score["length"] >= track_prefer_ratio * shape_score["length"]

    # Rescue path: if shape found nothing, allow weaker but persistent track.
    if (not use_track and method == "hybrid" and rescue_when_shape_missing
            and shape_score["length"] == 0 and track_score["length"] >= rescue_min_track_frames):
        if track_metrics.get("avg_score", 0.0) >= rescue_min_avg_score:
            use_track = True
            print(
                "  NOTE: Keyhole rescue activated (shape missing, "
                f"track_frames={track_score['length']}, "
                f"avg_score={track_metrics.get('avg_score', 0.0):.3f})."
            )

    if use_track and keyhole_raw_track:
        if track_metrics:
            print(f"  Keyhole selection: track (frames={track_score['length']}, "
                  f"leftward_px={track_score['leftward_px']:.1f}, "
                  f"vel_rms={track_metrics.get('velocity_rms', 0.0):.2f}, "
                  f"leftmost={track_metrics.get('leftmost_ratio', 0.0):.2f}, "
                  f"avg_score={track_metrics.get('avg_score', 0.0):.3f})")
        else:
            print(f"  Keyhole selection: track (frames={track_score['length']}, "
                  f"leftward_px={track_score['leftward_px']:.1f})")
        keyhole_raw, bubble_detections = split_keyhole_and_bubbles(
            all_detections, config, img_h, img_w, keyhole_raw_override=keyhole_raw_track,
        )
    else:
        print(f"  Keyhole selection: shape (frames={shape_score['length']}, "
              f"leftward_px={shape_score['leftward_px']:.1f})")
        keyhole_raw = keyhole_raw_shape
        bubble_detections = bubble_detections_shape

    # ── Optional: SAM point-prompt refinement of keyhole positions ─
    if config["detection"]["keyhole"].get("use_sam_refinement", False):
        print("\n--- Refining keyhole positions with SAM2 ---")
        if image_predictor is None:
            print("  WARNING: image_predictor not loaded (--skip-detection was set). "
                  "Re-run without --skip-detection to use SAM refinement.")
        else:
            keyhole_raw = refine_keyhole_with_sam(
                keyhole_raw, frame_names, frames_dir,
                image_predictor, device, config,
                crop_bottom_height=crop_bottom_height,
            )

    # ── Recover keyhole continuity across detector stalls ────────
    keyhole_before_recovery = len(keyhole_raw)
    keyhole_raw = recover_keyhole_track_with_template(
        keyhole_raw, all_detections, frame_names, frames_dir, config,
        crop_bottom_height=crop_bottom_height,
    )
    if len(keyhole_raw) != keyhole_before_recovery:
        print(f"  Keyhole recovery: {keyhole_before_recovery} -> {len(keyhole_raw)} frames")

    # ── Build keyhole trajectory ──────────────────────────────────
    print("\n--- Building keyhole trajectory ---")
    keyhole_positions, first_keyhole_frame = build_keyhole_trajectory(
        keyhole_raw, num_frames, config,
    )
    keyhole_positions = clamp_keyhole_positions_to_frame(keyhole_positions, img_w, img_h)
    observed_keyhole_frames = {
        fidx for fidx in keyhole_raw.keys() if 0 <= fidx < num_frames
    }

    trajectory_end_frame = None
    if first_keyhole_frame is None:
        print("  WARNING: No keyhole detected — all frames will be labeled 0.")
    else:
        stop_end_frame = find_keyhole_stop_frame(
            keyhole_positions, first_keyhole_frame, config,
        )
        loss_end_frame = estimate_end_frame_from_keyhole_loss(
            observed_keyhole_frames, num_frames, config,
        )

        end_candidates = [f for f in (stop_end_frame, loss_end_frame) if f is not None]
        if end_candidates:
            final_end_frame = min(end_candidates)
        else:
            final_end_frame = num_frames - 1

        validated_end_frame, validation_info = validate_and_adjust_end_with_post_end_evidence(
            final_end_frame, all_detections, num_frames, config, img_h, img_w,
        )
        if validated_end_frame > final_end_frame:
            final_end_frame = validated_end_frame
            if validation_info is not None:
                print(
                    "  NOTE: End validation extended trajectory "
                    f"to frame {validation_info['new_end']} "
                    f"(post-end evidence run {validation_info['run_start']}-{validation_info['run_end']}, "
                    f"len={validation_info['run_len']}, "
                    f"avg_score={validation_info['avg_score']:.3f}, "
                    f"median_cx={validation_info['median_cx']:.1f})."
                )

        if stop_end_frame is not None:
            print(f"  NOTE: Stop-rule end candidate: frame {stop_end_frame}.")
        if loss_end_frame is not None:
            print(f"  NOTE: Keyhole-loss end candidate: frame {loss_end_frame}.")

        if final_end_frame < num_frames - 1:
            print(f"  NOTE: Truncating trajectory at frame {final_end_frame}.")
            frame_names = frame_names[:final_end_frame + 1]
            num_frames = len(frame_names)
            keyhole_positions = {
                fidx: kh for fidx, kh in keyhole_positions.items()
                if fidx <= final_end_frame
            }
            observed_keyhole_frames = {
                fidx for fidx in observed_keyhole_frames if fidx <= final_end_frame
            }

        trajectory_end_frame = min(final_end_frame, num_frames - 1)

    # ── Remove invalid bubbles on the left side of keyhole ───────
    bubble_detections, removed_left = filter_bubbles_by_keyhole_side(
        bubble_detections, keyhole_positions, config,
        keyhole_observed_frames=observed_keyhole_frames,
        frame_width=img_w,
    )
    if removed_left > 0:
        print(f"  NOTE: Removed {removed_left} bubble detections left of keyhole.")

    # ── Build bubble tracks ───────────────────────────────────────
    print("\n--- Building bubble tracks ---")
    track_cfg    = config["tracking"]
    if first_keyhole_frame is None:
        bubble_detections_for_tracking = {}
        frames_list = []
        print("  NOTE: Ignoring all bubble detections (no keyhole detected).")
    else:
        bubble_detections_for_tracking = {
            fidx: det for fidx, det in bubble_detections.items()
            if fidx >= first_keyhole_frame and fidx < num_frames
        }
        frames_list = list(range(first_keyhole_frame, num_frames))
        if first_keyhole_frame > 0:
            print(f"  NOTE: Ignoring bubble detections before frame {first_keyhole_frame}.")

    bubble_tracks, _ = build_and_filter_tracks(
        bubble_detections_for_tracking, frames_list,
        track_iou_threshold = track_cfg["track_iou_threshold"],
        max_track_gap       = track_cfg["max_track_gap"],
        min_track_length    = track_cfg["bubble_min_track_length"],
    )
    print(f"  Bubble tracks after filtering: {len(bubble_tracks)}")

    # ── Merge fragmented bubble tracks ────────────────────────────
    print("\n--- Merging fragmented bubble tracks ---")
    bubble_tracks = merge_fragmented_tracks(bubble_tracks, config)

    # ── Classify bubble tracks ────────────────────────────────────
    print("\n--- Classifying bubble tracks ---")
    classified_tracks = classify_bubble_tracks(
        bubble_tracks, keyhole_positions, config,
    )

    # ── Label frames ──────────────────────────────────────────────
    print("\n--- Labeling frames ---")
    frame_labels = label_frames(
        num_frames, keyhole_positions, classified_tracks,
    )

    # ── Smooth labels ─────────────────────────────────────────────
    frame_labels = smooth_labels(frame_labels, config)

    # ── Group into intervals ──────────────────────────────────────
    intervals = group_into_intervals(frame_labels, num_frames, config)

    # ── Print summary ─────────────────────────────────────────────
    labels_cfg     = config["labels"]
    label_name_map  = {v["id"]: v["name"] for v in labels_cfg.values()}
    count_by_label  = Counter(frame_labels.values())

    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"First keyhole frame:    {first_keyhole_frame}")
    print(f"Trajectory end frame:   {trajectory_end_frame}")
    print(f"Keyhole detection:      GDINO (bubble.pore, shape split)")
    print(f"Total frames:           {num_frames}")
    print(f"Total intervals:        {len(intervals)}")
    print()
    for label_id in sorted(count_by_label.keys()):
        name  = label_name_map.get(label_id, "Unknown")
        count = count_by_label[label_id]
        pct   = 100.0 * count / num_frames
        print(f"  [{label_id}] {name}: {count} frames ({pct:.1f}%)")
    print()
    print("Intervals:")
    for iv in intervals:
        print(
            f"  Frames {iv['start_frame']:>5d}-{iv['end_frame']:>5d} "
            f"({iv['duration_frames']:>5d} frames): "
            f"[{iv['label_id']}] {iv['label_name']}"
        )

    # ── Generate labeled frames ───────────────────────────────────
    print("\n--- Generating labeled frames ---")
    generate_labeled_frames(
        frame_names, frames_dir, output_frames_dir,
        frame_labels, keyhole_positions, classified_tracks, config,
        crop_bottom_height=crop_bottom_height,
    )

    # ── Build output JSON ─────────────────────────────────────────
    frame_track_lookup = defaultdict(list)
    for track in classified_tracks:
        for entry in track["entries"]:
            frame_track_lookup[entry[0]].append(track)

    frame_label_details = []
    for fidx in range(num_frames):
        label_id   = frame_labels.get(fidx, 0)
        tracks_here = frame_track_lookup.get(fidx, [])
        detail = {
            "frame_index":   fidx,
            "frame_file":    frame_names[fidx] if fidx < len(frame_names) else None,
            "label_id":      label_id,
            "label_name":    label_name_map.get(label_id, "Unknown"),
            "keyhole_present": fidx in keyhole_positions,
            "num_bubbles":   len(tracks_here),
            "bubble_details": [
                {
                    "track_id":            t["id"],
                    "near_keyhole":        t["is_near_keyhole"],
                    "permanent":           t["is_permanent"],
                    "distance_to_keyhole": t["avg_distance_to_keyhole"],
                }
                for t in tracks_here
            ],
        }
        if fidx in keyhole_positions:
            kh = keyhole_positions[fidx]
            detail["keyhole_center"] = [round(kh["cx"], 2), round(kh["cy"], 2)]
        frame_label_details.append(detail)

    output = {
        "video_path":               video_path,
        "config_file":              args.config,
        "keyhole_detection_method": "gdino_shape_split",
        "first_keyhole_frame":      first_keyhole_frame,
        "trajectory_end_frame":     trajectory_end_frame,
        "image_size":               {"width": img_w, "height": img_h},
        "total_frames":             num_frames,
        "label_definitions": {
            k: {"id": v["id"], "name": v["name"], "description": v["description"]}
            for k, v in labels_cfg.items()
        },
        "summary": {
            label_name_map.get(lid, "Unknown"): count_by_label.get(lid, 0)
            for lid in range(5)
        },
        "intervals":  intervals,
        "tracks": {
            "bubble_tracks": [
                {
                    "track_id":                 t["id"],
                    "duration":                 t["duration"],
                    "is_near_keyhole":          t["is_near_keyhole"],
                    "is_permanent":             t["is_permanent"],
                    "avg_distance_to_keyhole":  t["avg_distance_to_keyhole"],
                    "first_frame":              t["entries"][0][0],
                    "last_frame":               t["entries"][-1][0],
                }
                for t in classified_tracks
            ],
        },
        "frame_labels": frame_label_details,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nLabeling results saved to {args.output}")


if __name__ == "__main__":
    main()
