"""
Evaluation script for Grounded-SAM bubble detection pipeline.

Usage:
  # Select random frames for labelme annotation:
    python evaluate.py --select-frames --num-frames 10

  # Evaluate pipeline masks against labelme ground truth:
    python evaluate.py

  # Both (select first, then evaluate existing annotations):
    python evaluate.py --select-frames --num-frames 5
"""

import os
import json
import random
import shutil
import argparse

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------- paths (match the pipeline) ----------
FRAMES_DIR = "./data/frames/custom_video_frames"
MASKS_DIR = "./data/output/masks"
LABELME_DIR = "./data/labelme"


def labelme_polygons_to_masks(json_path):
    """Convert labelme polygon annotations to binary instance masks."""
    with open(json_path, "r") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    masks = []

    for shape in data["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)

    if masks:
        return np.stack(masks)  # (N, H, W)
    return np.zeros((0, h, w), dtype=np.uint8)


def compute_iou_matrix(masks_a, masks_b):
    """Compute pairwise IoU between two sets of binary masks.

    Args:
        masks_a: (N, H, W) ground truth masks
        masks_b: (M, H, W) predicted masks

    Returns:
        iou_matrix: (N, M) pairwise IoU
    """
    n = masks_a.shape[0]
    m = masks_b.shape[0]
    iou_matrix = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            intersection = np.logical_and(masks_a[i], masks_b[j]).sum()
            union = np.logical_or(masks_a[i], masks_b[j]).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    return iou_matrix


def match_instances(gt_masks, pred_masks, iou_threshold=0.5):
    """Match predicted instances to GT using Hungarian algorithm.

    Returns:
        matched_ious: list of IoU values for matched pairs
        num_gt: total GT instances
        num_pred: total predicted instances
        num_matched: number of matches above iou_threshold
    """
    num_gt = gt_masks.shape[0]
    num_pred = pred_masks.shape[0]

    if num_gt == 0 or num_pred == 0:
        return [], num_gt, num_pred, 0

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    # Hungarian matching (maximize IoU = minimize negative IoU)
    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    num_matched = 0
    for gi, pi in zip(gt_idx, pred_idx):
        iou_val = iou_matrix[gi, pi]
        matched_ious.append(iou_val)
        if iou_val >= iou_threshold:
            num_matched += 1

    return matched_ious, num_gt, num_pred, num_matched


def pixel_level_metrics(gt_masks, pred_masks):
    """Compute pixel-level precision, recall, F1, and IoU.

    Treats all instances as a single binary foreground mask.
    """
    gt_binary = gt_masks.any(axis=0).astype(np.uint8) if gt_masks.shape[0] > 0 else np.zeros(gt_masks.shape[1:], dtype=np.uint8)
    pred_binary = pred_masks.any(axis=0).astype(np.uint8) if pred_masks.shape[0] > 0 else np.zeros_like(gt_binary)

    tp = np.logical_and(gt_binary, pred_binary).sum()
    fp = np.logical_and(~gt_binary.astype(bool), pred_binary.astype(bool)).sum()
    fn = np.logical_and(gt_binary.astype(bool), ~pred_binary.astype(bool)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0

    return precision, recall, f1, iou


def evaluate_frame(gt_masks, pred_masks, iou_threshold=0.5):
    """Evaluate a single frame. Returns a dict of metrics."""
    # Pixel-level
    precision, recall, f1, pixel_iou = pixel_level_metrics(gt_masks, pred_masks)

    # Instance-level
    matched_ious, num_gt, num_pred, num_matched = match_instances(
        gt_masks, pred_masks, iou_threshold
    )
    mean_matched_iou = np.mean(matched_ious) if matched_ious else 0.0
    inst_precision = num_matched / num_pred if num_pred > 0 else 0.0
    inst_recall = num_matched / num_gt if num_gt > 0 else 0.0
    inst_f1 = (
        2 * inst_precision * inst_recall / (inst_precision + inst_recall)
        if (inst_precision + inst_recall) > 0
        else 0.0
    )

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": pixel_iou,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "num_matched": num_matched,
        "inst_precision": inst_precision,
        "inst_recall": inst_recall,
        "inst_f1": inst_f1,
        "mean_matched_iou": mean_matched_iou,
    }


def select_random_frames(num_frames, seed=42):
    """Select random frames from the pipeline output and copy to labelme dir."""
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith(".npy")])
    if not mask_files:
        print(f"No mask files found in {MASKS_DIR}. Run the pipeline first.")
        return

    # Exclude frames that already have labelme annotations
    existing = {os.path.splitext(f)[0] for f in os.listdir(LABELME_DIR) if f.endswith(".json")}
    available = [f for f in mask_files if os.path.splitext(f)[0] not in existing]

    if len(available) < num_frames:
        print(f"Only {len(available)} unannotated frames available (requested {num_frames}).")
        num_frames = len(available)

    random.seed(seed)
    selected = random.sample(available, num_frames)
    os.makedirs(LABELME_DIR, exist_ok=True)

    print(f"Selected {num_frames} frames for labelme annotation:")
    for mask_file in sorted(selected):
        frame_id = os.path.splitext(mask_file)[0]
        src_img = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
        dst_img = os.path.join(LABELME_DIR, f"{frame_id}.jpg")

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            print(f"  {frame_id}.jpg -> {LABELME_DIR}/")
        else:
            print(f"  WARNING: {src_img} not found, skipping.")

    print(f"\nNow run:  labelme {LABELME_DIR}")
    print("Annotate bubbles as polygons with label 'bubble', then run this script again to evaluate.")


def run_evaluation(iou_threshold=0.5):
    """Evaluate pipeline predictions against all labelme annotations."""
    json_files = sorted([f for f in os.listdir(LABELME_DIR) if f.endswith(".json")])
    if not json_files:
        print(f"No labelme annotations found in {LABELME_DIR}.")
        return

    all_results = []
    print(f"{'Frame':<12} {'GT':>4} {'Pred':>5} {'Match':>6} "
          f"{'Px-IoU':>7} {'Px-P':>6} {'Px-R':>6} {'Px-F1':>6} "
          f"{'In-P':>6} {'In-R':>6} {'In-F1':>6} {'mIoU':>6}")
    print("-" * 95)

    for json_file in json_files:
        frame_id = os.path.splitext(json_file)[0]
        json_path = os.path.join(LABELME_DIR, json_file)
        mask_path = os.path.join(MASKS_DIR, f"{frame_id}.npy")

        # Load GT
        gt_masks = labelme_polygons_to_masks(json_path)

        # Load predictions
        if os.path.exists(mask_path):
            pred_masks = np.load(mask_path)
            if pred_masks.ndim == 2:
                pred_masks = pred_masks[np.newaxis]
        else:
            print(f"  WARNING: {mask_path} not found, using empty prediction.")
            h, w = gt_masks.shape[1], gt_masks.shape[2]
            pred_masks = np.zeros((0, h, w), dtype=np.uint8)

        metrics = evaluate_frame(gt_masks, pred_masks, iou_threshold)
        metrics["frame_id"] = frame_id
        all_results.append(metrics)

        print(f"{frame_id:<12} {metrics['num_gt']:>4} {metrics['num_pred']:>5} {metrics['num_matched']:>6} "
              f"{metrics['pixel_iou']:>7.3f} {metrics['pixel_precision']:>6.3f} {metrics['pixel_recall']:>6.3f} {metrics['pixel_f1']:>6.3f} "
              f"{metrics['inst_precision']:>6.3f} {metrics['inst_recall']:>6.3f} {metrics['inst_f1']:>6.3f} {metrics['mean_matched_iou']:>6.3f}")

    # Summary
    print("-" * 95)
    n = len(all_results)
    avg = lambda key: sum(r[key] for r in all_results) / n

    total_gt = sum(r["num_gt"] for r in all_results)
    total_pred = sum(r["num_pred"] for r in all_results)
    total_matched = sum(r["num_matched"] for r in all_results)

    print(f"{'AVERAGE':<12} {total_gt/n:>4.0f} {total_pred/n:>5.0f} {total_matched/n:>6.0f} "
          f"{avg('pixel_iou'):>7.3f} {avg('pixel_precision'):>6.3f} {avg('pixel_recall'):>6.3f} {avg('pixel_f1'):>6.3f} "
          f"{avg('inst_precision'):>6.3f} {avg('inst_recall'):>6.3f} {avg('inst_f1'):>6.3f} {avg('mean_matched_iou'):>6.3f}")

    print(f"\nTotal frames evaluated: {n}")
    print(f"Total GT bubbles: {total_gt},  Total predicted: {total_pred},  Total matched (IoU>={iou_threshold}): {total_matched}")
    overall_inst_p = total_matched / total_pred if total_pred > 0 else 0
    overall_inst_r = total_matched / total_gt if total_gt > 0 else 0
    overall_inst_f1 = 2 * overall_inst_p * overall_inst_r / (overall_inst_p + overall_inst_r) if (overall_inst_p + overall_inst_r) > 0 else 0
    print(f"Overall instance precision: {overall_inst_p:.3f},  recall: {overall_inst_r:.3f},  F1: {overall_inst_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bubble detection pipeline")
    parser.add_argument("--select-frames", action="store_true",
                        help="Select random frames and copy to labelme dir for annotation")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of random frames to select (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for frame selection (default: 42)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for instance matching (default: 0.5)")
    parser.add_argument("--masks-dir", type=str, default=MASKS_DIR,
                        help="Directory containing predicted mask .npy files")
    parser.add_argument("--labelme-dir", type=str, default=LABELME_DIR,
                        help="Directory containing labelme annotations")
    parser.add_argument("--frames-dir", type=str, default=FRAMES_DIR,
                        help="Directory containing raw frame images")

    args = parser.parse_args()
    MASKS_DIR = args.masks_dir
    LABELME_DIR = args.labelme_dir
    FRAMES_DIR = args.frames_dir

    if args.select_frames:
        select_random_frames(args.num_frames, seed=args.seed)

    run_evaluation(iou_threshold=args.iou_threshold)
