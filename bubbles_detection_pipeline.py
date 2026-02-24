# Grounded-SAM-2 bubble detection pipeline
# Uses local Grounding DINO (HuggingFace transformers) + SAM2 image predictor
# Per-frame detection: no video tracking, fresh detections on every frame

import os
import json
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.video_utils import create_video_from_images

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

"""
Hyperparameters
"""
VIDEO_PATH = "./data/raw/x_ray_video.mp4"
TEXT_PROMPT = "bubble."
OUTPUT_VIDEO_PATH = "./data/output/bubbles_groundedSAM.mp4"
SOURCE_VIDEO_FRAME_DIR = "./data/frames/custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./data/output/tracking_results"
SAVE_MASKS_DIR = "./data/output/masks"
OUTPUT_JSON_PATH = "./data/output/detections.json"
BOX_THRESHOLD = 0.2
MAX_BOX_AREA_RATIO = 0.05  # filter out boxes larger than 5% of image area
DETECT_INTERVAL = 1  # run detection every N frames (1 = every frame)

# Grounding DINO model from HuggingFace (auto-downloads weights on first run)
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"  # or "IDEA-Research/grounding-dino-tiny"

"""
Step 1: Environment settings and model initialization
"""
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Grounding DINO locally
print(f"Loading Grounding DINO from {GDINO_MODEL_ID}...")
gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
print("Grounding DINO loaded.")

# Load SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

"""
Step 2: Extract video frames
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames,
    overwrite=True,
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# Crop each saved frame to keep only the bottom 310 rows
CROP_TOP = 800 - 310
for fname in tqdm(os.listdir(source_frames), desc="Cropping Frames"):
    fpath = os.path.join(source_frames, fname)
    if os.path.splitext(fname)[-1].lower() in [".jpg", ".jpeg"]:
        img = cv2.imread(fpath)
        cropped = img[CROP_TOP:, :, :]
        cv2.imwrite(fpath, cropped)

frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
num_frames = len(frame_names)

# get image dimensions for box area filtering
sample_image = Image.open(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[0]))
img_w, img_h = sample_image.size
img_area = img_h * img_w

"""
Step 3: Local detection helper
"""


def detect_on_frame(frame_idx):
    """Run local Grounding DINO on a single frame, return filtered boxes, class names, and scores."""
    fpath = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
    image = Image.open(fpath).convert("RGB")

    inputs = gdino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        target_sizes=[(img_h, img_w)],
    )[0]

    boxes, names, conf_scores = [], [], []
    for bbox, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        bbox = bbox.cpu().tolist()  # [x1, y1, x2, y2] in pixels
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if box_area / img_area > MAX_BOX_AREA_RATIO:
            continue
        boxes.append(bbox)
        names.append(label)
        conf_scores.append(round(float(score.cpu()), 4))
    return boxes, names, conf_scores


"""
Step 4: Find the first frame with detections
"""
NUM_PROBE_FRAMES = 10
COARSE_STEP = max(1, num_frames // NUM_PROBE_FRAMES)
print(f"Scanning for first frame with detections (step={COARSE_STEP})...")

start_frame_idx = None

for pidx in range(0, num_frames, COARSE_STEP):
    boxes, _, _ = detect_on_frame(pidx)
    print(f"  Frame {pidx:>5d}: {len(boxes)} detections")
    if len(boxes) > 0:
        start_frame_idx = pidx
        break

if start_frame_idx is None:
    boxes, _, _ = detect_on_frame(num_frames - 1)
    if len(boxes) > 0:
        start_frame_idx = num_frames - 1

if start_frame_idx is None:
    raise RuntimeError("No detections found on any probed frame. Try lowering BOX_THRESHOLD.")

# Refine with binary search
lo = max(0, start_frame_idx - COARSE_STEP)
hi = start_frame_idx
while lo < hi:
    mid = (lo + hi) // 2
    boxes, _, _ = detect_on_frame(mid)
    print(f"  Refine frame {mid:>5d}: {len(boxes)} detections")
    if len(boxes) > 0:
        hi = mid
    else:
        lo = mid + 1

start_frame_idx = lo
print(f"\nFirst frame with detections: {start_frame_idx}")
print(f"Will process frames {start_frame_idx}-{num_frames - 1} (every {DETECT_INTERVAL} frames)")

"""
Step 5: Per-frame detection + SAM2 segmentation
"""
os.makedirs(SAVE_MASKS_DIR, exist_ok=True)
os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

all_frame_detections = []

frames_to_process = range(start_frame_idx, num_frames, DETECT_INTERVAL)
for frame_idx in tqdm(frames_to_process, desc="Detect + Segment"):
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
    img_bgr = cv2.imread(img_path)

    # --- Local Grounding DINO detection ---
    input_boxes, class_names, conf_scores = detect_on_frame(frame_idx)

    frame_record = {
        "frame_index": frame_idx,
        "frame_file": frame_names[frame_idx],
        "num_bubbles": len(input_boxes),
        "bubbles": [],
    }

    if len(input_boxes) == 0:
        frame_masks = np.zeros((0, img_h, img_w), dtype=np.uint8)
        np.save(os.path.join(SAVE_MASKS_DIR, f"{frame_idx:05d}.npy"), frame_masks)
        cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), img_bgr)
        all_frame_detections.append(frame_record)
        continue

    # --- SAM2 segmentation ---
    image_rgb = np.array(Image.open(img_path).convert("RGB"))
    image_predictor.set_image(image_rgb)

    boxes_np = np.array(input_boxes)
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # ensure masks are boolean for supervision annotators
    masks = masks.astype(bool)

    # --- Collect per-bubble info ---
    for i in range(len(input_boxes)):
        bbox = input_boxes[i]
        mask_area = int(masks[i].sum())
        bubble_info = {
            "id": i,
            "label": class_names[i],
            "confidence": conf_scores[i],
            "bbox": [round(c, 2) for c in bbox],
            "bbox_area": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 2),
            "mask_area_pixels": mask_area,
        }
        frame_record["bubbles"].append(bubble_info)

    all_frame_detections.append(frame_record)

    # --- Save raw masks ---
    np.save(os.path.join(SAVE_MASKS_DIR, f"{frame_idx:05d}.npy"), masks.astype(np.uint8))

    # --- Visualize ---
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        class_id=np.arange(len(masks), dtype=np.int32),
    )
    annotated = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=class_names)
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated)

"""
Step 6: Save detection results as JSON
"""
total_bubbles = sum(f["num_bubbles"] for f in all_frame_detections)
frames_with_detections = sum(1 for f in all_frame_detections if f["num_bubbles"] > 0)

detection_output = {
    "video_path": VIDEO_PATH,
    "text_prompt": TEXT_PROMPT,
    "model": GDINO_MODEL_ID,
    "box_threshold": BOX_THRESHOLD,
    "max_box_area_ratio": MAX_BOX_AREA_RATIO,
    "image_size": {"width": img_w, "height": img_h},
    "total_frames_processed": len(all_frame_detections),
    "frames_with_detections": frames_with_detections,
    "total_bubbles_detected": total_bubbles,
    "start_frame_index": start_frame_idx,
    "detect_interval": DETECT_INTERVAL,
    "frames": all_frame_detections,
}

os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(detection_output, f, indent=2)
print(f"\nDetection results saved to {OUTPUT_JSON_PATH}")

"""
Step 7: Convert annotated frames to video
"""
create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
