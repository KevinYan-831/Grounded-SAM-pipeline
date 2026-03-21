"""
Shared detection and tracking utilities for the bubble detection pipeline.

Extracted from bubbles_detection_pipeline.py and tune_params.py to avoid
code duplication across the detection, tuning, and labeling pipelines.
"""

import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_models(gdino_model_id, sam2_checkpoint, sam2_model_cfg):
    """
    Load Grounding DINO and SAM2 models.

    Returns:
        (gdino_processor, gdino_model, image_predictor, device)
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Grounding DINO from {gdino_model_id}...")
    gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_model_id
    ).to(device)
    print("Grounding DINO loaded.")

    print(f"Loading SAM2 from {sam2_checkpoint}...")
    sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    print("SAM2 loaded.")

    return gdino_processor, gdino_model, image_predictor, device


def extract_and_crop_frames(video_path, output_dir, crop_top=None):
    """
    Extract video frames and optionally crop them.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save frames.
        crop_top: If set, number of pixels to remove from top of each frame.

    Returns:
        (frame_names, img_w, img_h) — sorted list of frame filenames and image dimensions.
    """
    video_info = sv.VideoInfo.from_video_path(video_path)
    print(video_info)
    frame_generator = sv.get_video_frames_generator(
        video_path, stride=1, start=0, end=None
    )

    source_frames = Path(output_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames,
        overwrite=True,
        image_name_pattern="{:05d}.jpg",
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    if crop_top is not None:
        for fname in tqdm(os.listdir(source_frames), desc="Cropping Frames"):
            fpath = os.path.join(source_frames, fname)
            if os.path.splitext(fname)[-1].lower() in [".jpg", ".jpeg"]:
                img = cv2.imread(fpath)
                cropped = img[crop_top:, :, :]
                cv2.imwrite(fpath, cropped)

    frame_names = [
        p
        for p in os.listdir(output_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    sample_image = Image.open(os.path.join(output_dir, frame_names[0]))
    img_w, img_h = sample_image.size

    return frame_names, img_w, img_h


def detect_on_frame(
    frame_idx,
    frame_names,
    source_dir,
    gdino_processor,
    gdino_model,
    text_prompt,
    box_threshold,
    img_h,
    img_w,
    max_box_area_ratio,
    device,
):
    """
    Run Grounding DINO on a single frame with the given text prompt.

    Returns:
        (boxes, names, scores) — lists of detections passing the area filter.
    """
    img_area = img_h * img_w
    fpath = os.path.join(source_dir, frame_names[frame_idx])
    image = Image.open(fpath).convert("RGB")

    inputs = gdino_processor(
        images=image, text=text_prompt, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        target_sizes=[(img_h, img_w)],
    )[0]

    boxes, names, conf_scores = [], [], []
    for bbox, score, label in zip(
        results["boxes"], results["scores"], results["text_labels"]
    ):
        bbox = bbox.cpu().tolist()
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if box_area / img_area > max_box_area_ratio:
            continue
        boxes.append(bbox)
        names.append(label)
        conf_scores.append(round(float(score.cpu()), 4))

    # NMS: remove duplicate detections of the same object
    boxes, names, conf_scores = _nms(boxes, names, conf_scores, iou_threshold=0.5)
    return boxes, names, conf_scores


def _nms(boxes, names, scores, iou_threshold=0.5):
    """Non-Maximum Suppression: remove overlapping detections, keep highest score."""
    if len(boxes) <= 1:
        return boxes, names, scores
    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx in suppressed or jdx == idx:
                continue
            if box_iou(boxes[idx], boxes[jdx]) > iou_threshold:
                suppressed.add(jdx)
    return (
        [boxes[i] for i in keep],
        [names[i] for i in keep],
        [scores[i] for i in keep],
    )


def box_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
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


def build_and_filter_tracks(
    detections, frames_list, track_iou_threshold, max_track_gap,
    min_track_length, detect_interval=1,
):
    """
    Build IoU-based tracks and filter short ones.

    Each track entry is a tuple: (frame_idx, box, name, score).

    Returns:
        (valid_tracks, per_frame_dict)
        - valid_tracks: list of track dicts with 'id', 'entries', 'last_frame', 'last_box'
        - per_frame_dict: {frame_idx: [(box, name, score, track_id), ...]}
    """
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
                track["entries"].append(
                    (frame_idx, boxes[di], names[di], scores[di])
                )
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

    valid_tracks = [
        t for t in finished_tracks if len(t["entries"]) >= min_track_length
    ]

    # Build per-frame lookup: frame_idx -> [(box, name, score, track_id), ...]
    per_frame = defaultdict(list)
    for track in valid_tracks:
        for (fidx, box, name, score) in track["entries"]:
            per_frame[fidx].append((box, name, score, track["id"]))

    return valid_tracks, dict(per_frame)
