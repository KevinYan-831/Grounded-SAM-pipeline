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
import cv2
import numpy as np

from collections import Counter, defaultdict
from tqdm import tqdm

from utils.detection_utils import (
    load_models,
    extract_and_crop_frames,
    detect_on_frame,
    build_and_filter_tracks,
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

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ─────────────────────────────────────────────────────────────
# Detection: run GDINO once per frame with wide thresholds
# ─────────────────────────────────────────────────────────────
def run_all_detections(
    frame_names, source_dir, config, gdino_processor, gdino_model,
    img_h, img_w, device,
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

    text_prompt = bb_cfg["text_prompt"]
    detection_threshold = min(bb_cfg["box_threshold"], kh_cfg["box_threshold"])
    detection_area_ratio = max(bb_cfg["max_box_area_ratio"], kh_cfg["max_box_area_ratio"])

    all_detections = {}
    for fidx in tqdm(range(len(frame_names)), desc="Detection"):
        boxes, names, scores = detect_on_frame(
            fidx, frame_names, source_dir, gdino_processor, gdino_model,
            text_prompt, detection_threshold,
            img_h, img_w, detection_area_ratio, device,
        )
        all_detections[fidx] = (boxes, names, scores)

    total = sum(len(v[0]) for v in all_detections.values())
    frames_with = sum(1 for v in all_detections.values() if len(v[0]) > 0)
    print(f"  Total raw detections: {total} across {frames_with} frames")
    return all_detections


# ─────────────────────────────────────────────────────────────
# Split detections: leftmost tall object → keyhole, rest → bubble
# ─────────────────────────────────────────────────────────────
def split_keyhole_and_bubbles(all_detections, config, img_h, img_w):
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

    keyhole_raw       = {}
    bubble_detections = {}

    for fidx, (boxes, names, scores) in all_detections.items():
        # --- find keyhole: leftmost detection passing all shape constraints ---
        kh_best    = None
        kh_best_x1 = float("inf")

        for box, name, score in zip(boxes, names, scores):
            x1, y1, x2, y2 = box
            h_px = y2 - y1
            w_px = x2 - x1
            ar   = h_px / w_px if w_px > 0 else 0.0

            if (h_px >= min_h and
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

        # --- bubbles: everything that is NOT the keyhole box ---
        bb_boxes, bb_names, bb_scores = [], [], []
        for box, name, score in zip(boxes, names, scores):
            x1, y1, x2, y2 = box
            # Skip the box we picked as keyhole
            if kh_best is not None and abs(x1 - kh_best["x"]) < 0.5 and abs(y1 - kh_best["y"]) < 0.5:
                continue
            if score >= bb_threshold:
                h_px = y2 - y1
                w_px = x2 - x1
                if (h_px * w_px) / img_area <= bb_max_area:
                    bb_boxes.append(box)
                    bb_names.append(name)
                    bb_scores.append(score)

        if kh_best is not None:
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
    from PIL import Image as PILImage

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
        img_pil = PILImage.open(fpath).convert("RGB")
        img_np  = np.array(img_pil)

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

    print(f"  Raw keyhole candidates: {len(keyhole_raw)} frames")

    # Temporal filter — reject cx outliers
    filtered = _temporal_filter(keyhole_raw, max_jump, median_window)
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
    positions = _interpolate_positions(filtered, first_kh, num_frames - 1)
    print(f"  After interpolation: {len(positions)} frames")

    # Smooth
    positions = _smooth_positions(positions, smooth_window)

    return dict(positions), first_kh


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
        img   = cv2.imread(fpath)
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
        "--skip-extraction", action="store_true",
        help="Skip frame extraction (reuse already-extracted frames)",
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Skip GDINO detection and reuse cached raw_detections.json",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_JSON_PATH,
        help=f"Output JSON path (default: {OUTPUT_JSON_PATH})",
    )
    args = parser.parse_args()

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
        from PIL import Image as PILImage

        frame_names = sorted(
            [p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
             if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")],
            key=lambda p: int(os.path.splitext(p)[0]),
        )
        sample  = PILImage.open(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[0]))
        img_w, img_h = sample.size
        print(f"Reusing {len(frame_names)} frames from {SOURCE_VIDEO_FRAME_DIR}")
    else:
        frame_names, img_w, img_h = extract_and_crop_frames(
            VIDEO_PATH, SOURCE_VIDEO_FRAME_DIR, crop_top=CROP_TOP,
        )

    num_frames = len(frame_names)
    print(f"Total frames: {num_frames}, image size: {img_w}x{img_h}")

    # ── GDINO detection ──────────────────────────────────────────
    print("\n--- Running GDINO detection ---")
    if args.skip_detection:
        print(f"  Loading from cache: {DETECTION_CACHE_PATH}")
        all_detections = load_detection_cache(DETECTION_CACHE_PATH)
    else:
        all_detections = run_all_detections(
            frame_names, SOURCE_VIDEO_FRAME_DIR, config,
            gdino_processor, gdino_model, img_h, img_w, device,
        )
        save_detection_cache(all_detections, DETECTION_CACHE_PATH)

    # ── Split detections: keyhole vs bubbles ─────────────────────
    print("\n--- Splitting detections by shape ---")
    keyhole_raw, bubble_detections = split_keyhole_and_bubbles(
        all_detections, config, img_h, img_w,
    )

    # ── Optional: SAM point-prompt refinement of keyhole positions ─
    if config["detection"]["keyhole"].get("use_sam_refinement", False):
        print("\n--- Refining keyhole positions with SAM2 ---")
        if image_predictor is None:
            print("  WARNING: image_predictor not loaded (--skip-detection was set). "
                  "Re-run without --skip-detection to use SAM refinement.")
        else:
            keyhole_raw = refine_keyhole_with_sam(
                keyhole_raw, frame_names, SOURCE_VIDEO_FRAME_DIR,
                image_predictor, device, config,
            )

    # ── Build keyhole trajectory ──────────────────────────────────
    print("\n--- Building keyhole trajectory ---")
    keyhole_positions, first_keyhole_frame = build_keyhole_trajectory(
        keyhole_raw, num_frames, config,
    )

    if first_keyhole_frame is None:
        print("  WARNING: No keyhole detected — all frames will be labeled 0.")

    # ── Build bubble tracks ───────────────────────────────────────
    print("\n--- Building bubble tracks ---")
    track_cfg    = config["tracking"]
    frames_list  = list(range(num_frames))

    bubble_tracks, _ = build_and_filter_tracks(
        bubble_detections, frames_list,
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
        frame_names, SOURCE_VIDEO_FRAME_DIR, OUTPUT_FRAMES_DIR,
        frame_labels, keyhole_positions, classified_tracks, config,
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
        "video_path":               VIDEO_PATH,
        "config_file":              args.config,
        "keyhole_detection_method": "gdino_shape_split",
        "first_keyhole_frame":      first_keyhole_frame,
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
