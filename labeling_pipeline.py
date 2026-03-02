"""
Frame labeling pipeline for bubble detection in X-ray video.

Assigns per-frame state labels based on keyhole presence, bubble proximity
to the keyhole, and track persistence. Outputs a labeled interval dataset
as JSON.

Labels:
  0 - No Signal
  1 - Normal Process
  2 - Unstable Process without Pore Generation
  3 - Transient Pore Generation
  4 - Permanent Pore Generation

Usage:
    python labeling_pipeline.py
    python labeling_pipeline.py --config custom_rules.yaml
    python labeling_pipeline.py --skip-extraction   # reuse already-extracted frames
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
    box_iou,
    build_and_filter_tracks,
)
from utils.keyhole_detector import detect_keyhole_all_frames

# ─────────────────────────────────────────────────────────────
# Paths (same as detection pipeline)
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = "./data/raw/x_ray_video.mp4"
SOURCE_VIDEO_FRAME_DIR = "./data/frames/custom_video_frames"
OUTPUT_JSON_PATH = "./data/labeling/labeling_results.json"
OUTPUT_FRAMES_DIR = "./data/labeling/frames"
CROP_TOP = 800 - 310  # keep bottom 310 rows

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ─────────────────────────────────────────────────────────────
# Bubble detection (Grounding DINO)
# ─────────────────────────────────────────────────────────────
def run_detection(
    frame_names, source_dir, config, gdino_processor, gdino_model,
    img_h, img_w, device,
):
    """
    Run Grounding DINO once per frame with the bubble text prompt.
    All objects (keyholes + bubbles) are detected together.

    Returns:
        all_detections: {frame_idx: (boxes, names, scores)}
    """
    bb_cfg = config["detection"]["bubble"]
    all_detections = {}

    for fidx in tqdm(range(len(frame_names)), desc="Detection"):
        boxes, names, scores = detect_on_frame(
            fidx, frame_names, source_dir, gdino_processor, gdino_model,
            bb_cfg["text_prompt"], bb_cfg["box_threshold"],
            img_h, img_w, bb_cfg["max_box_area_ratio"], device,
        )
        all_detections[fidx] = (boxes, names, scores)

    total = sum(len(v[0]) for v in all_detections.values())
    frames_with = sum(1 for v in all_detections.values() if len(v[0]) > 0)
    print(f"Total: {total} detections across {frames_with} frames")
    return all_detections


# (Keyhole detection is now handled by utils/keyhole_detector.py)


# ─────────────────────────────────────────────────────────────
# Classify bubble tracks by proximity & persistence
# ─────────────────────────────────────────────────────────────
def classify_bubble_tracks(bubble_tracks, keyhole_positions, config):
    """
    For each bubble track, compute:
      - average distance to keyhole center
      - near/far classification
      - transient/permanent classification

    Returns:
        List of track dicts with added classification metadata.
    """
    prox_cfg = config["proximity"]
    track_cfg = config["track_classification"]

    near_threshold = prox_cfg["near_threshold_pixels"]
    transient_max = track_cfg["transient_max_frames"]
    permanent_min = track_cfg["permanent_min_frames"]
    method = prox_cfg.get("method", "center_distance")

    classified = []
    for track in bubble_tracks:
        entries = track["entries"]
        duration = len(entries)

        distances = []
        for entry in entries:
            fidx, box = entry[0], entry[1]
            bubble_cx = (box[0] + box[2]) / 2
            bubble_cy = (box[1] + box[3]) / 2

            if fidx in keyhole_positions:
                kh = keyhole_positions[fidx]
                kh_cx = kh["cx"]
                kh_cy = kh["cy"]
                kh_box = kh["bbox"]
                if method == "edge_distance":
                    dx = max(0, max(box[0] - kh_box[2], kh_box[0] - box[2]))
                    dy = max(0, max(box[1] - kh_box[3], kh_box[1] - box[3]))
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                else:  # center_distance (default)
                    dist = np.sqrt(
                        (bubble_cx - kh_cx) ** 2 + (bubble_cy - kh_cy) ** 2
                    )
                distances.append(dist)

        avg_distance = float(np.mean(distances)) if distances else float("inf")
        is_near = avg_distance <= near_threshold
        is_permanent = duration >= permanent_min
        is_transient = duration < transient_max

        classified.append({
            **track,
            "duration": duration,
            "avg_distance_to_keyhole": round(avg_distance, 2),
            "is_near_keyhole": is_near,
            "is_permanent": is_permanent,
            "is_transient": is_transient,
        })

    near_count = sum(1 for t in classified if t["is_near_keyhole"])
    perm_count = sum(1 for t in classified if t["is_permanent"])
    print(
        f"Bubble tracks: {len(classified)} total, "
        f"{near_count} near keyhole, {perm_count} permanent"
    )
    return classified


# ─────────────────────────────────────────────────────────────
# Per-frame labeling
# ─────────────────────────────────────────────────────────────
def label_frames(
    num_frames, first_keyhole_frame, keyhole_positions,
    classified_bubble_tracks, config,
):
    """
    Assign a label to each frame using the updated rules:

      0 - No Signal:       No keyhole detected in the frame.
      1 - Normal Process:  Entire trajectory has NO permanent pore anywhere
                           → all keyhole frames are normal.
      2 - Unstable:        Trajectory HAS permanent pores, but this specific
                           frame has no pore.
      3 - Transient Pore:  A pore exists at this frame that will disappear.
      4 - Permanent Pore:  A pore exists at this frame that stays permanently.

    The logic is:
      1. First, determine globally whether ANY permanent pore track exists.
      2. If none → all keyhole frames get label 1 (Normal Process).
      3. If yes  → per-frame decision based on what pores are present.
    """
    # Build per-frame bubble track lookup
    frame_tracks = defaultdict(list)
    for track in classified_bubble_tracks:
        for entry in track["entries"]:
            frame_tracks[entry[0]].append(track)

    # Global check: does ANY permanent pore exist in the entire trajectory?
    has_any_permanent = any(t["is_permanent"] for t in classified_bubble_tracks)

    frame_labels = {}
    for fidx in range(num_frames):
        has_keyhole = fidx in keyhole_positions
        tracks_here = frame_tracks.get(fidx, [])
        has_pores = len(tracks_here) > 0

        # Rule 0: No keyhole → No Signal
        if not has_keyhole:
            frame_labels[fidx] = 0
            continue

        # Keyhole is present from here on

        # Rule 1: No permanent pore in entire trajectory → Normal Process
        if not has_any_permanent:
            frame_labels[fidx] = 1
            continue

        # Trajectory HAS permanent pores somewhere

        # Rule 2: This frame has no pore → Unstable
        if not has_pores:
            frame_labels[fidx] = 2
            continue

        # This frame has pore(s) — classify by track type
        has_permanent_here = any(t["is_permanent"] for t in tracks_here)

        if has_permanent_here:
            # Rule 4: Permanent pore at this frame
            frame_labels[fidx] = 4
        else:
            # Rule 3: Transient pore at this frame
            frame_labels[fidx] = 3

    return frame_labels


# ─────────────────────────────────────────────────────────────
# Label smoothing (majority vote)
# ─────────────────────────────────────────────────────────────
def smooth_labels(frame_labels, num_frames, config):
    """Apply majority-vote smoothing within a sliding window."""
    window = config["intervals"].get("smoothing_window", 1)
    if window <= 1:
        return frame_labels

    sorted_frames = sorted(frame_labels.keys())
    labels_array = [frame_labels[f] for f in sorted_frames]
    smoothed = {}

    for i, fidx in enumerate(sorted_frames):
        start = max(0, i - window // 2)
        end = min(len(sorted_frames), i + window // 2 + 1)
        window_labels = labels_array[start:end]
        counts = Counter(window_labels)
        # On tie, keep current label
        most_common = counts.most_common()
        top_count = most_common[0][1]
        candidates = [lbl for lbl, cnt in most_common if cnt == top_count]
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
    labels_cfg = config["labels"]
    label_name_map = {v["id"]: v["name"] for v in labels_cfg.values()}

    intervals = []
    current_label = None
    current_start = None

    for fidx in range(num_frames):
        label = frame_labels.get(fidx, 0)
        if label != current_label:
            if current_label is not None:
                intervals.append({
                    "start_frame": current_start,
                    "end_frame": fidx - 1,
                    "duration_frames": fidx - current_start,
                    "label_id": current_label,
                    "label_name": label_name_map.get(current_label, "Unknown"),
                })
            current_label = label
            current_start = fidx

    # Final interval
    if current_label is not None:
        intervals.append({
            "start_frame": current_start,
            "end_frame": num_frames - 1,
            "duration_frames": num_frames - current_start,
            "label_id": current_label,
            "label_name": label_name_map.get(current_label, "Unknown"),
        })

    # Optionally merge short intervals into neighbors
    min_len = config["intervals"].get("min_interval_length", 1)
    if min_len > 1:
        intervals = _merge_short_intervals(intervals, min_len)

    return intervals


def _merge_short_intervals(intervals, min_length):
    """Merge intervals shorter than min_length into their longest neighbor."""
    if len(intervals) <= 1:
        return intervals

    merged = list(intervals)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            iv = merged[i]
            if iv["duration_frames"] < min_length and len(new_merged) > 0:
                # Merge into previous interval
                new_merged[-1]["end_frame"] = iv["end_frame"]
                new_merged[-1]["duration_frames"] = (
                    new_merged[-1]["end_frame"] - new_merged[-1]["start_frame"] + 1
                )
                changed = True
            elif (
                iv["duration_frames"] < min_length
                and i + 1 < len(merged)
            ):
                # Merge into next interval
                merged[i + 1]["start_frame"] = iv["start_frame"]
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
    Save annotated frames with a colored label bar, keyhole box, and
    bubble boxes drawn on each frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_cfg = config["labels"]
    label_name_map = {v["id"]: v["name"] for v in labels_cfg.values()}
    label_color_map = {v["id"]: tuple(v["color"]) for v in labels_cfg.values()}

    # Build per-frame bubble box lookup
    frame_bubble_boxes = defaultdict(list)
    for track in classified_bubble_tracks:
        for entry in track["entries"]:
            fidx, box = entry[0], entry[1]
            frame_bubble_boxes[fidx].append((box, track))

    for fidx in tqdm(range(len(frame_names)), desc="Saving labeled frames"):
        fpath = os.path.join(source_dir, frame_names[fidx])
        img = cv2.imread(fpath)
        if img is None:
            continue

        label_id = frame_labels.get(fidx, 0)
        label_name = label_name_map.get(label_id, "Unknown")
        color_rgb = label_color_map.get(label_id, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        h, w = img.shape[:2]

        # Draw colored label bar at top
        bar_h = 30
        cv2.rectangle(img, (0, 0), (w, bar_h), color_bgr, -1)
        cv2.putText(
            img, f"[{label_id}] {label_name}  (frame {fidx})",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

        # Draw keyhole box (white)
        if fidx in keyhole_positions:
            kh = keyhole_positions[fidx]
            kh_box = kh["bbox"]
            pt1 = (int(kh_box[0]), int(kh_box[1]))
            pt2 = (int(kh_box[2]), int(kh_box[3]))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
            cv2.putText(
                img, "keyhole", (pt1[0], pt1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

        # Draw bubble boxes
        for box, track in frame_bubble_boxes.get(fidx, []):
            if track["is_near_keyhole"]:
                bb_color = (0, 0, 255) if track["is_permanent"] else (0, 165, 255)
            else:
                bb_color = (0, 255, 255)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, bb_color, 1)
            lbl = f"T{track['id']}"
            cv2.putText(
                img, lbl, (pt1[0], pt1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, bb_color, 1,
            )

        cv2.imwrite(os.path.join(output_dir, f"{fidx:05d}.jpg"), img)

    print(f"Labeled frames saved to {output_dir}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Label video frames based on keyhole and bubble detection"
    )
    parser.add_argument(
        "--config", type=str, default="labeling_rules.yaml",
        help="Path to labeling rules YAML config (default: labeling_rules.yaml)",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip frame extraction (reuse already-extracted frames)",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_JSON_PATH,
        help=f"Output JSON path (default: {OUTPUT_JSON_PATH})",
    )
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded labeling config from {args.config}")

    # ── Load models ──
    gdino_processor, gdino_model, image_predictor, device = load_models(
        GDINO_MODEL_ID, SAM2_CHECKPOINT, SAM2_MODEL_CFG,
    )

    # ── Extract frames ──
    if args.skip_extraction:
        from PIL import Image as PILImage

        frame_names = [
            p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        sample = PILImage.open(
            os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[0])
        )
        img_w, img_h = sample.size
        print(f"Reusing {len(frame_names)} frames from {SOURCE_VIDEO_FRAME_DIR}")
    else:
        frame_names, img_w, img_h = extract_and_crop_frames(
            VIDEO_PATH, SOURCE_VIDEO_FRAME_DIR, crop_top=CROP_TOP,
        )

    num_frames = len(frame_names)
    print(f"Total frames: {num_frames}, image size: {img_w}x{img_h}")

    # ── Keyhole detection (classical CV) ──
    print("\n--- Detecting keyhole (classical CV) ---")
    keyhole_positions, first_keyhole_frame = detect_keyhole_all_frames(
        SOURCE_VIDEO_FRAME_DIR, frame_names, config,
    )
    kh_method = "classical_cv"

    # ── Bubble detection (Grounding DINO) ──
    print("\n--- Running bubble detection ---")
    all_detections = run_detection(
        frame_names, SOURCE_VIDEO_FRAME_DIR, config,
        gdino_processor, gdino_model, img_h, img_w, device,
    )

    # ── Build bubble tracks ──
    print("\n--- Building bubble tracks ---")
    track_cfg = config["tracking"]
    frames_list = list(range(num_frames))

    bubble_tracks, bb_per_frame = build_and_filter_tracks(
        all_detections, frames_list,
        track_iou_threshold=track_cfg["track_iou_threshold"],
        max_track_gap=track_cfg["max_track_gap"],
        min_track_length=track_cfg["bubble_min_track_length"],
    )
    print(f"Bubble tracks: {len(bubble_tracks)}")

    # ── Classify bubble tracks ──
    print("\n--- Classifying bubble tracks ---")
    classified_tracks = classify_bubble_tracks(
        bubble_tracks, keyhole_positions, config,
    )

    # ── Label frames ──
    print("\n--- Labeling frames ---")
    frame_labels = label_frames(
        num_frames, first_keyhole_frame, keyhole_positions,
        classified_tracks, config,
    )

    # ── Smooth labels ──
    frame_labels = smooth_labels(frame_labels, num_frames, config)

    # ── Group into intervals ──
    intervals = group_into_intervals(frame_labels, num_frames, config)

    # ── Print summary ──
    labels_cfg = config["labels"]
    label_name_map = {v["id"]: v["name"] for v in labels_cfg.values()}
    count_by_label = Counter(frame_labels.values())

    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"First keyhole frame: {first_keyhole_frame}")
    print(f"Keyhole detection method: {kh_method}")
    print(f"Total frames: {num_frames}")
    print(f"Total intervals: {len(intervals)}")
    print()
    for label_id in sorted(count_by_label.keys()):
        name = label_name_map.get(label_id, "Unknown")
        count = count_by_label[label_id]
        pct = 100.0 * count / num_frames
        print(f"  [{label_id}] {name}: {count} frames ({pct:.1f}%)")
    print()
    print("Intervals:")
    for iv in intervals:
        print(
            f"  Frames {iv['start_frame']:>5d}-{iv['end_frame']:>5d} "
            f"({iv['duration_frames']:>5d} frames): "
            f"[{iv['label_id']}] {iv['label_name']}"
        )

    # ── Generate labeled frames ──
    print("\n--- Generating labeled frames ---")
    generate_labeled_frames(
        frame_names, SOURCE_VIDEO_FRAME_DIR, OUTPUT_FRAMES_DIR,
        frame_labels, keyhole_positions, classified_tracks, config,
    )

    # ── Build per-frame detail for output ──
    frame_track_lookup = defaultdict(list)
    for track in classified_tracks:
        for entry in track["entries"]:
            frame_track_lookup[entry[0]].append(track)

    frame_label_details = []
    for fidx in range(num_frames):
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
        frame_label_details.append(detail)

    # ── Build output JSON ──
    output = {
        "video_path": VIDEO_PATH,
        "config_file": args.config,
        "keyhole_detection_method": kh_method,
        "first_keyhole_frame": first_keyhole_frame,
        "image_size": {"width": img_w, "height": img_h},
        "total_frames": num_frames,
        "label_definitions": {
            k: {"id": v["id"], "name": v["name"], "description": v["description"]}
            for k, v in labels_cfg.items()
        },
        "summary": {
            label_name_map.get(lid, "Unknown"): count_by_label.get(lid, 0)
            for lid in range(5)
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
            ],
        },
        "frame_labels": frame_label_details,
    }

    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nLabeling results saved to {args.output}")


if __name__ == "__main__":
    main()
