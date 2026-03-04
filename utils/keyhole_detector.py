"""
Keyhole trajectory post-processing utilities.

After GDINO detects sparse keyhole positions (via the leftmost-tall-object
heuristic in labeling_pipeline.py), these functions clean up the raw
detections into a smooth, gap-free trajectory:

  1. _temporal_filter       — reject cx outliers using a local median filter
  2. _find_first_keyhole_frame — locate the first reliable keyhole appearance
  3. _interpolate_positions  — linearly fill gaps; extrapolate to end of video
  4. _smooth_positions       — rolling-average smoothing of cx / cy
"""

from collections import OrderedDict


def _temporal_filter(raw_detections, max_jump_px=120, median_window=30):
    """
    Apply temporal consistency filtering:
      1. Compute local median of cx over a sliding window.
      2. Reject detections whose cx deviates > max_jump_px from that median.

    Args:
        raw_detections: {frame_idx: dict} where each dict has at least "cx".
        max_jump_px:    Maximum allowed deviation from local median (pixels).
        median_window:  Half-width of the local median window (frames).

    Returns:
        Filtered dict with the same structure as raw_detections.
    """
    if not raw_detections:
        return {}

    frames    = sorted(raw_detections.keys())
    cx_values = {f: raw_detections[f]["cx"] for f in frames}

    filtered = {}
    for i, fidx in enumerate(frames):
        nearby_cx   = [
            cx_values[frames[j]]
            for j in range(max(0, i - median_window), min(len(frames), i + median_window + 1))
        ]
        local_median = sorted(nearby_cx)[len(nearby_cx) // 2]
        if abs(cx_values[fidx] - local_median) <= max_jump_px:
            filtered[fidx] = raw_detections[fidx]

    return filtered


def _find_first_keyhole_frame(filtered, min_x=100):
    """
    Identify the first reliable keyhole frame.

    The keyhole starts on the right side of the image (high cx) and moves
    left over time.  Early false positives tend to cluster near the left
    edge (cx < min_x); this filter skips them.

    Args:
        filtered: output of _temporal_filter.
        min_x:    Minimum cx for a detection to count as the real keyhole.

    Returns:
        Frame index of the first real keyhole, or None.
    """
    if not filtered:
        return None
    for fidx in sorted(filtered.keys()):
        if filtered[fidx]["cx"] > min_x:
            return fidx
    return None


def _interpolate_positions(filtered, first_frame, last_frame):
    """
    Linearly interpolate keyhole position for frames between first_frame
    and last_frame that are missing from filtered detections.
    Extrapolates at constant position after the last known detection.

    Args:
        filtered:    {frame_idx: dict(x, y, w, h, cx, cy, ...)}
        first_frame: first frame to include in the output
        last_frame:  last frame to include (inclusive)

    Returns:
        OrderedDict: {frame_idx: {"cx", "cy", "bbox": [x1,y1,x2,y2]}}
    """
    positions = OrderedDict()
    frames    = sorted(f for f in filtered if first_frame <= f <= last_frame)

    if not frames:
        return positions

    # Known detections
    for fidx in frames:
        c    = filtered[fidx]
        bbox = [c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]]
        positions[fidx] = {"cx": c["cx"], "cy": c["cy"], "bbox": bbox}

    # Interpolate gaps between known frames
    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end   = frames[i + 1]
        gap     = f_end - f_start - 1
        if gap <= 0:
            continue

        c1 = filtered[f_start]
        c2 = filtered[f_end]
        for j in range(1, gap + 1):
            t  = j / (gap + 1)
            cx = c1["cx"] + t * (c2["cx"] - c1["cx"])
            cy = c1["cy"] + t * (c2["cy"] - c1["cy"])
            x1 = c1["x"] + t * (c2["x"] - c1["x"])
            y1 = c1["y"] + t * (c2["y"] - c1["y"])
            x2 = (c1["x"] + c1["w"]) + t * ((c2["x"] + c2["w"]) - (c1["x"] + c1["w"]))
            y2 = (c1["y"] + c1["h"]) + t * ((c2["y"] + c2["h"]) - (c1["y"] + c1["h"]))
            positions[f_start + j] = {"cx": cx, "cy": cy, "bbox": [x1, y1, x2, y2]}

    # Extrapolate at constant position after the last detection
    if last_frame > frames[-1]:
        last_c = filtered[frames[-1]]
        bbox   = [last_c["x"], last_c["y"],
                  last_c["x"] + last_c["w"], last_c["y"] + last_c["h"]]
        for fidx in range(frames[-1] + 1, last_frame + 1):
            positions[fidx] = {"cx": last_c["cx"], "cy": last_c["cy"], "bbox": bbox}

    return OrderedDict(sorted(positions.items()))


def _smooth_positions(positions, window=5):
    """
    Apply rolling-average smoothing to cx and cy.

    Args:
        positions: OrderedDict from _interpolate_positions.
        window:    Number of frames in the rolling average.

    Returns:
        Smoothed OrderedDict with same keys.
    """
    if not positions or window <= 1:
        return positions

    frames   = sorted(positions.keys())
    smoothed = OrderedDict()
    for i, fidx in enumerate(frames):
        start    = max(0, i - window // 2)
        end      = min(len(frames), i + window // 2 + 1)
        win      = frames[start:end]
        cx_avg   = sum(positions[f]["cx"] for f in win) / len(win)
        cy_avg   = sum(positions[f]["cy"] for f in win) / len(win)
        smoothed[fidx] = {
            "cx": cx_avg, "cy": cy_avg,
            "bbox": positions[fidx]["bbox"],
        }
    return smoothed
