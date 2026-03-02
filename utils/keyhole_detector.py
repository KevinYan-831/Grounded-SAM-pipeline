"""
Classical-CV keyhole detector for X-ray welding video frames.

The keyhole appears as a dark vertical feature extending from the top of
the image.  Detection is based on local background subtraction (large
Gaussian blur), thresholding, and contour analysis.  A temporal consistency
filter rejects spurious detections and fills short gaps via interpolation.

This module is independent of Grounding DINO / SAM2 and only requires
OpenCV + NumPy.
"""

import cv2
import numpy as np
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────
# Single-frame detection
# ─────────────────────────────────────────────────────────────
def _detect_candidates(img_gray, sigma, top_margin, min_height, min_ar, edge_margin):
    """Return a list of keyhole candidate dicts for one frame."""
    h, w = img_gray.shape
    img_f = img_gray.astype(np.float32)

    # Local background via large Gaussian blur
    bg = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)

    # Positive diff → pixel darker than local background
    diff = bg - img_f

    # Zero out edges (sensor artifacts)
    diff[:, :edge_margin] = 0
    diff[:, w - edge_margin:] = 0

    pos_vals = diff[diff > 0]
    if len(pos_vals) == 0:
        return []
    thresh = pos_vals.mean() + 2.0 * pos_vals.std()

    binary = (diff > thresh).astype(np.uint8) * 255

    # Vertical closing to bridge vertical gaps in the keyhole stem
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vert_kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 30 or ch < min_height or cw == 0:
            continue
        ar = ch / cw
        if ar < min_ar:
            continue

        touches_top = y <= top_margin
        # Strongly prefer features connected to the image top
        score = ch * ar * (5.0 if touches_top else 0.3)

        candidates.append({
            "x": x, "y": y, "w": cw, "h": ch,
            "cx": x + cw / 2.0, "cy": y + ch / 2.0,
            "ar": ar, "area": int(area),
            "score": score, "top": touches_top,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────
# Temporal consistency filtering
# ─────────────────────────────────────────────────────────────
def _temporal_filter(raw_detections, max_jump_px=120, median_window=30):
    """
    Apply temporal consistency filtering:
      1. For top-connected detections, apply local median filter on cx.
      2. Reject detections whose cx deviates > max_jump_px from local median.

    Returns filtered dict: {frame_idx: candidate_dict}.
    """
    if not raw_detections:
        return {}

    frames = sorted(raw_detections.keys())
    cx_values = {f: raw_detections[f]["cx"] for f in frames}

    filtered = {}
    for i, fidx in enumerate(frames):
        # Gather nearby cx values for local median
        nearby_cx = []
        for j in range(max(0, i - median_window), min(len(frames), i + median_window + 1)):
            nearby_cx.append(cx_values[frames[j]])
        local_median = sorted(nearby_cx)[len(nearby_cx) // 2]

        if abs(cx_values[fidx] - local_median) <= max_jump_px:
            filtered[fidx] = raw_detections[fidx]

    return filtered


def _find_first_keyhole_frame(filtered, min_x=200):
    """
    Identify the first real keyhole frame.

    Early false positives tend to be near the left edge (x < min_x).
    The real keyhole starts at the right side of the image and moves left.
    """
    if not filtered:
        return None
    for fidx in sorted(filtered.keys()):
        if filtered[fidx]["cx"] > min_x:
            return fidx
    return None


# ─────────────────────────────────────────────────────────────
# Gap interpolation
# ─────────────────────────────────────────────────────────────
def _interpolate_positions(filtered, first_frame, last_frame):
    """
    Linearly interpolate keyhole position for frames between first_frame
    and last_frame that are missing from filtered detections.

    Returns OrderedDict: {frame_idx: {"cx": ..., "cy": ..., "bbox": [x1,y1,x2,y2]}}.
    """
    positions = OrderedDict()
    frames = sorted(f for f in filtered if first_frame <= f <= last_frame)

    if not frames:
        return positions

    # Add known detections
    for fidx in frames:
        c = filtered[fidx]
        bbox = [c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]]
        positions[fidx] = {"cx": c["cx"], "cy": c["cy"], "bbox": bbox}

    # Interpolate gaps
    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end = frames[i + 1]
        gap = f_end - f_start - 1
        if gap <= 0:
            continue

        c1 = filtered[f_start]
        c2 = filtered[f_end]
        for j in range(1, gap + 1):
            t = j / (gap + 1)
            cx = c1["cx"] + t * (c2["cx"] - c1["cx"])
            cy = c1["cy"] + t * (c2["cy"] - c1["cy"])
            # Interpolate bounding box too
            x1 = c1["x"] + t * (c2["x"] - c1["x"])
            y1 = c1["y"] + t * (c2["y"] - c1["y"])
            x2 = (c1["x"] + c1["w"]) + t * ((c2["x"] + c2["w"]) - (c1["x"] + c1["w"]))
            y2 = (c1["y"] + c1["h"]) + t * ((c2["y"] + c2["h"]) - (c1["y"] + c1["h"]))
            fidx_interp = f_start + j
            positions[fidx_interp] = {
                "cx": cx, "cy": cy,
                "bbox": [x1, y1, x2, y2],
            }

    # Extrapolate after last detection to end of video (constant position)
    if last_frame > frames[-1]:
        last_c = filtered[frames[-1]]
        bbox = [last_c["x"], last_c["y"],
                last_c["x"] + last_c["w"], last_c["y"] + last_c["h"]]
        for fidx in range(frames[-1] + 1, last_frame + 1):
            positions[fidx] = {"cx": last_c["cx"], "cy": last_c["cy"], "bbox": bbox}

    return OrderedDict(sorted(positions.items()))


# ─────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────
def _smooth_positions(positions, window=5):
    """Apply rolling average smoothing to cx, cy positions."""
    if not positions or window <= 1:
        return positions

    frames = sorted(positions.keys())
    smoothed = OrderedDict()
    for i, fidx in enumerate(frames):
        start = max(0, i - window // 2)
        end = min(len(frames), i + window // 2 + 1)
        win_frames = frames[start:end]
        cx_avg = sum(positions[f]["cx"] for f in win_frames) / len(win_frames)
        cy_avg = sum(positions[f]["cy"] for f in win_frames) / len(win_frames)
        smoothed[fidx] = {
            "cx": cx_avg, "cy": cy_avg,
            "bbox": positions[fidx]["bbox"],
        }
    return smoothed


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def detect_keyhole_all_frames(
    frame_dir,
    frame_names,
    config,
):
    """
    Detect the keyhole in every frame using classical CV.

    Args:
        frame_dir:   Directory containing the extracted frame images.
        frame_names: Sorted list of frame filenames.
        config:      The full labeling_rules config dict.

    Returns:
        (keyhole_positions, first_keyhole_frame)
        - keyhole_positions: {frame_idx: {"cx", "cy", "bbox"}} for all
          frames from first keyhole appearance to end of video.
        - first_keyhole_frame: int or None
    """
    kh_cfg = config.get("keyhole_cv", {})
    sigma = kh_cfg.get("gaussian_sigma", 80)
    top_margin = kh_cfg.get("top_margin", 15)
    min_height = kh_cfg.get("min_height", 40)
    min_ar = kh_cfg.get("min_aspect_ratio", 1.5)
    edge_margin = kh_cfg.get("edge_margin", 50)
    max_jump = kh_cfg.get("max_jump_px", 120)
    median_window = kh_cfg.get("median_window", 30)
    smooth_window = kh_cfg.get("smoothing_window", 5)
    first_frame_min_x = kh_cfg.get("first_frame_min_x", 200)

    from tqdm import tqdm

    num_frames = len(frame_names)

    # Step 1: Detect on every frame
    raw_top = {}  # frame_idx -> best top-connected candidate
    for fidx in tqdm(range(num_frames), desc="Keyhole detection (CV)"):
        fpath = f"{frame_dir}/{frame_names[fidx]}"
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        candidates = _detect_candidates(
            img, sigma, top_margin, min_height, min_ar, edge_margin,
        )
        # Pick the best top-connected candidate
        top_cands = [c for c in candidates if c["top"]]
        if top_cands:
            raw_top[fidx] = top_cands[0]

    print(f"  Raw top-connected detections: {len(raw_top)} / {num_frames} frames")

    # Step 2: Temporal consistency filter
    filtered = _temporal_filter(raw_top, max_jump, median_window)
    print(f"  After temporal filter: {len(filtered)} frames")

    # Step 3: Find first real keyhole frame
    first_kh = _find_first_keyhole_frame(filtered, first_frame_min_x)
    if first_kh is None:
        print("  WARNING: No keyhole detected in any frame.")
        return {}, None

    # Remove detections before first real keyhole
    filtered = {f: c for f, c in filtered.items() if f >= first_kh}
    last_kh = max(filtered.keys())
    print(f"  Keyhole range: frames {first_kh} - {last_kh}")

    # Step 4: Interpolate gaps (extend to end of video)
    positions = _interpolate_positions(filtered, first_kh, num_frames - 1)
    print(f"  After interpolation: {len(positions)} frames")

    # Step 5: Smooth
    positions = _smooth_positions(positions, smooth_window)

    return positions, first_kh
