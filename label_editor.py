#!/usr/bin/env python3
"""
Gradio app for manually reviewing and adjusting frame labels.
Shows clean frames (from xray-enhanced-frames) side by side with
annotated reference frames (from labeling_trajectories_manual/frames)
where available. Covers ALL frames in each trajectory.
"""

import json
import os
import glob

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

RESULTS_ROOT = "/home/jixin/labeling_trajectories_manual"
CLEAN_FRAMES_ROOT = "/home/jixin/xray-enhanced-frames"

LABEL_NAMES = {
    0: "Normal Process",
    1: "Unstable Process without Pore Generation",
    2: "Permanent Pore Generation",
    3: "Transient Pore Generation",
    4: "No Signal",
}
LABEL_SHORT = {0: "Normal", 1: "Unstable", 2: "Permanent", 3: "Transient", 4: "No Signal"}
LABEL_COLORS = {
    0: (76, 175, 80),     # green
    1: (255, 193, 7),     # amber
    2: (244, 67, 54),     # red
    3: (33, 150, 243),    # blue
    4: (158, 158, 158),   # gray
}


def find_trajectories():
    results = []
    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, "**/labeling_results.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), RESULTS_ROOT)
        results.append(rel)
    return results


def load_trajectory(traj_id):
    result_path = os.path.join(RESULTS_ROOT, traj_id, "labeling_results.json")
    with open(result_path) as f:
        return json.load(f)


def save_trajectory(traj_id, data):
    result_path = os.path.join(RESULTS_ROOT, traj_id, "labeling_results.json")
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)


def add_label_banner(img, label_id, frame_idx, title=""):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.copy()
    draw = ImageDraw.Draw(img)
    color = LABEL_COLORS.get(label_id, (128, 128, 128))
    banner_h = 32
    draw.rectangle([0, 0, img.width, banner_h], fill=color)
    text = f"Frame {frame_idx} | {LABEL_SHORT.get(label_id, '?')}"
    if title:
        text = f"{title} | {text}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    brightness = sum(color) / 3
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw.text((8, 6), text, fill=text_color, font=font)
    return img


def get_clean_frame(traj_id, frame_idx):
    clean_dir = os.path.join(CLEAN_FRAMES_ROOT, traj_id)
    for ext in ["png", "jpg"]:
        path = os.path.join(clean_dir, f"{frame_idx:05d}.{ext}")
        if os.path.exists(path):
            return Image.open(path)
    return None


def get_reference_frame(traj_id, frame_idx):
    frames_dir = os.path.join(RESULTS_ROOT, traj_id, "frames")
    for ext in ["jpg", "png"]:
        path = os.path.join(frames_dir, f"{frame_idx:05d}.{ext}")
        if os.path.exists(path):
            return Image.open(path)
    return None


def expand_frame_labels(data, traj_id):
    """Expand frame_labels to cover ALL frames (0..total_frames-1).

    Previously labeled frames keep their labels.
    Frames outside the analysis range default to label 4 (No Signal).
    """
    total_frames = data.get("total_frames", 0)
    if total_frames == 0:
        clean_dir = os.path.join(CLEAN_FRAMES_ROOT, traj_id)
        if os.path.isdir(clean_dir):
            total_frames = len([f for f in os.listdir(clean_dir)
                               if f.endswith((".png", ".jpg"))])
        data["total_frames"] = total_frames

    existing = {fl["frame_index"]: fl for fl in data.get("frame_labels", [])}
    analysis_range = data.get("analysis_frame_range", [None, None])

    all_labels = []
    for i in range(total_frames):
        if i in existing:
            all_labels.append(existing[i])
        else:
            # Default: No Signal for frames outside analysis range
            all_labels.append({
                "frame_index": i,
                "frame_file": f"{i:05d}.png",
                "label_id": 4,
                "label_name": LABEL_NAMES[4],
                "keyhole_present": False,
                "num_bubbles": 0,
                "bubble_details": [],
            })

    data["frame_labels"] = all_labels
    return data


# ── Global state ──
current_data = {}
current_traj = ""
current_ref_frames = set()
unsaved_changes = set()


def _build_summary(traj_id, data):
    frame_labels = data.get("frame_labels", [])
    total = len(frame_labels)
    analysis_range = data.get("analysis_frame_range", [None, None])

    text = f"Trajectory: {traj_id}\n"
    text += f"Total frames: {total}\n"
    if analysis_range[0] is not None:
        text += f"Analysis range: {analysis_range[0]} - {analysis_range[1]}\n"
    text += "\nLabel counts:\n"
    counts = {}
    for name in LABEL_NAMES.values():
        counts[name] = 0
    for fl in frame_labels:
        counts[LABEL_NAMES.get(fl["label_id"], "Unknown")] = counts.get(
            LABEL_NAMES.get(fl["label_id"], "Unknown"), 0) + 1
    for name, count in counts.items():
        text += f"  {name}: {count}\n"
    return text


def on_select_trajectory(traj_id):
    global current_data, current_traj, current_ref_frames
    if not traj_id:
        return None, None, "", 0, gr.update(maximum=0), ""

    current_traj = traj_id
    current_data = load_trajectory(traj_id)
    current_data = expand_frame_labels(current_data, traj_id)

    # Detect which frames have reference images
    ref_dir = os.path.join(RESULTS_ROOT, traj_id, "frames")
    current_ref_frames = set()
    if os.path.isdir(ref_dir):
        for f in os.listdir(ref_dir):
            name, ext = os.path.splitext(f)
            if ext in (".jpg", ".png"):
                try:
                    current_ref_frames.add(int(name))
                except ValueError:
                    pass

    frame_labels = current_data["frame_labels"]
    n = len(frame_labels)
    summary_text = _build_summary(traj_id, current_data)

    fl = frame_labels[0]
    clean_img = get_clean_frame(traj_id, fl["frame_index"])
    if clean_img:
        clean_img = add_label_banner(clean_img, fl["label_id"], fl["frame_index"], "CLEAN")

    ref_img = None
    if fl["frame_index"] in current_ref_frames:
        ref_img = get_reference_frame(traj_id, fl["frame_index"])
        if ref_img:
            ref_img = add_label_banner(ref_img, fl["label_id"], fl["frame_index"], "REFERENCE")

    return clean_img, ref_img, summary_text, 0, gr.update(maximum=n - 1), LABEL_SHORT[fl["label_id"]]


def on_frame_change(frame_slider):
    if not current_data or not current_traj:
        return None, None, ""

    frame_labels = current_data.get("frame_labels", [])
    idx = int(frame_slider)
    if idx < 0 or idx >= len(frame_labels):
        return None, None, ""

    fl = frame_labels[idx]
    clean_img = get_clean_frame(current_traj, fl["frame_index"])
    if clean_img:
        clean_img = add_label_banner(clean_img, fl["label_id"], fl["frame_index"], "CLEAN")

    ref_img = None
    if fl["frame_index"] in current_ref_frames:
        ref_img = get_reference_frame(current_traj, fl["frame_index"])
        if ref_img:
            ref_img = add_label_banner(ref_img, fl["label_id"], fl["frame_index"], "REFERENCE")

    return clean_img, ref_img, LABEL_SHORT.get(fl["label_id"], "?")


def _get_both_images(idx):
    frame_labels = current_data.get("frame_labels", [])
    fl = frame_labels[idx]
    clean_img = get_clean_frame(current_traj, fl["frame_index"])
    if clean_img:
        clean_img = add_label_banner(clean_img, fl["label_id"], fl["frame_index"], "CLEAN")

    ref_img = None
    if fl["frame_index"] in current_ref_frames:
        ref_img = get_reference_frame(current_traj, fl["frame_index"])
        if ref_img:
            ref_img = add_label_banner(ref_img, fl["label_id"], fl["frame_index"], "REFERENCE")

    return clean_img, ref_img


def on_label_change(frame_slider, new_label):
    global unsaved_changes
    if not current_data or not current_traj:
        return None, None, "No trajectory loaded"

    label_map = {v: k for k, v in LABEL_SHORT.items()}
    label_id = label_map.get(new_label)
    if label_id is None:
        return None, None, "Invalid label"

    frame_labels = current_data.get("frame_labels", [])
    idx = int(frame_slider)
    if idx < 0 or idx >= len(frame_labels):
        return None, None, "Invalid frame index"

    old_label = frame_labels[idx]["label_id"]
    if old_label == label_id:
        clean_img, ref_img = _get_both_images(idx)
        return clean_img, ref_img, "No change"

    frame_labels[idx]["label_id"] = label_id
    frame_labels[idx]["label_name"] = LABEL_NAMES[label_id]
    unsaved_changes.add(current_traj)

    clean_img, ref_img = _get_both_images(idx)
    fi = frame_labels[idx]["frame_index"]
    return clean_img, ref_img, f"Frame {fi}: {LABEL_SHORT[old_label]} -> {LABEL_SHORT[label_id]} (unsaved)"


def on_batch_label(start_frame, end_frame, new_label):
    global unsaved_changes
    if not current_data or not current_traj:
        return "No trajectory loaded"

    label_map = {v: k for k, v in LABEL_SHORT.items()}
    label_id = label_map.get(new_label)
    if label_id is None:
        return "Invalid label"

    frame_labels = current_data.get("frame_labels", [])
    count = 0
    for fl in frame_labels:
        if start_frame <= fl["frame_index"] <= end_frame:
            fl["label_id"] = label_id
            fl["label_name"] = LABEL_NAMES[label_id]
            count += 1

    if count > 0:
        unsaved_changes.add(current_traj)
        return f"Updated {count} frames ({int(start_frame)}-{int(end_frame)}) to {LABEL_SHORT[label_id]} (unsaved)"
    return "No frames in range"


def on_save():
    global unsaved_changes
    if not current_data or not current_traj:
        return "No trajectory loaded"

    if current_traj not in unsaved_changes:
        return "No changes to save"

    frame_labels = current_data["frame_labels"]
    total_frames = current_data.get("total_frames", len(frame_labels))

    # Recompute label_sequence (all frames)
    full_seq = [4] * total_frames  # default No Signal
    for fl in frame_labels:
        fi = fl["frame_index"]
        if fi < total_frames:
            full_seq[fi] = fl["label_id"]
    current_data["label_sequence"] = full_seq

    # analyzed_label_sequence = just the analysis range
    analysis_range = current_data.get("analysis_frame_range", [0, total_frames - 1])
    start, end = analysis_range
    current_data["analyzed_label_sequence"] = full_seq[start:end + 1]

    # Recompute summary (all frames)
    summary = {}
    for name in LABEL_NAMES.values():
        summary[name] = 0
    for fl in frame_labels:
        lid = fl["label_id"]
        name = LABEL_NAMES.get(lid, "Unknown")
        summary[name] = summary.get(name, 0) + 1
    current_data["summary"] = summary

    # Recompute transition counts (all frames)
    n_labels = len(LABEL_NAMES)
    tc = [[0] * n_labels for _ in range(n_labels)]
    for i in range(len(full_seq) - 1):
        src, dst = full_seq[i], full_seq[i + 1]
        if 0 <= src < n_labels and 0 <= dst < n_labels:
            tc[src][dst] += 1
    current_data["transition_counts"] = tc

    # Recompute intervals
    current_data["intervals"] = recompute_intervals(frame_labels)

    save_trajectory(current_traj, current_data)
    unsaved_changes.discard(current_traj)

    return _build_summary(current_traj, current_data) + "\nSaved successfully!"


def recompute_intervals(frame_labels):
    if not frame_labels:
        return []
    intervals = []
    current_label = frame_labels[0]["label_id"]
    start_frame = frame_labels[0]["frame_index"]
    prev_frame = start_frame

    for fl in frame_labels[1:]:
        if fl["label_id"] != current_label:
            intervals.append({
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "num_frames": prev_frame - start_frame + 1,
                "label_id": current_label,
                "label_name": LABEL_NAMES.get(current_label, "Unknown"),
            })
            current_label = fl["label_id"]
            start_frame = fl["frame_index"]
        prev_frame = fl["frame_index"]

    intervals.append({
        "start_frame": start_frame,
        "end_frame": prev_frame,
        "num_frames": prev_frame - start_frame + 1,
        "label_id": current_label,
        "label_name": LABEL_NAMES.get(current_label, "Unknown"),
    })
    return intervals


def on_prev(frame_slider):
    return max(0, int(frame_slider) - 1)


def on_next(frame_slider):
    n = len(current_data.get("frame_labels", [1]))
    return min(n - 1, int(frame_slider) + 1)


def get_label_timeline():
    if not current_data or not current_traj:
        return None
    frame_labels = current_data.get("frame_labels", [])
    if not frame_labels:
        return None

    w = max(len(frame_labels), 800)
    h = 80
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    bar_w = w / len(frame_labels)
    for i, fl in enumerate(frame_labels):
        color = LABEL_COLORS.get(fl["label_id"], (128, 128, 128))
        x0 = int(i * bar_w)
        x1 = int((i + 1) * bar_w)
        draw.rectangle([x0, 0, x1, h - 25], fill=color)

    # Mark analysis range
    analysis_range = current_data.get("analysis_frame_range", [None, None])
    if analysis_range[0] is not None:
        a_start = int(analysis_range[0] * bar_w)
        a_end = int((analysis_range[1] + 1) * bar_w)
        draw.rectangle([a_start, h - 25, a_end, h - 22], fill=(0, 0, 0))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    x_offset = 5
    for lid in sorted(LABEL_SHORT.keys()):
        color = LABEL_COLORS[lid]
        draw.rectangle([x_offset, h - 18, x_offset + 12, h - 6], fill=color, outline=(0, 0, 0))
        draw.text((x_offset + 15, h - 18), LABEL_SHORT[lid], fill=(0, 0, 0), font=font)
        x_offset += 90
    draw.rectangle([x_offset, h - 18, x_offset + 12, h - 6], fill=(0, 0, 0))
    draw.text((x_offset + 15, h - 18), "Analysis range", fill=(0, 0, 0), font=font)

    return img


def on_refresh_timeline():
    return get_label_timeline()


# ── Build UI ──
trajectories = find_trajectories()

with gr.Blocks(title="Label Editor") as app:
    gr.Markdown("# Frame Label Editor")
    gr.Markdown("Clean frame (left) | Reference with detections (right, when available)")

    with gr.Row():
        traj_dropdown = gr.Dropdown(
            choices=trajectories, label="Trajectory", scale=3,
            info=f"{len(trajectories)} trajectories available"
        )
        save_btn = gr.Button("Save Changes", variant="primary", scale=1)

    summary_box = gr.Textbox(label="Summary", lines=9, interactive=False)

    timeline_img = gr.Image(label="Label Timeline (black bar = analysis range)", type="pil", height=100)

    with gr.Row():
        prev_btn = gr.Button("< Prev", scale=1)
        frame_slider = gr.Slider(0, 0, value=0, step=1, label="Frame", scale=6)
        next_btn = gr.Button("Next >", scale=1)

    with gr.Row():
        clean_img = gr.Image(label="Clean Frame", type="pil", height=400)
        ref_img = gr.Image(label="Reference (Detection Overlay)", type="pil", height=400)

    with gr.Row():
        current_label = gr.Textbox(label="Current Label", interactive=False, scale=2)
        new_label = gr.Radio(
            choices=list(LABEL_SHORT.values()),
            label="Set Label",
            scale=3,
        )

    change_status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### Batch Label (apply to frame index range)")
    with gr.Row():
        batch_start = gr.Number(label="Start Frame Index", precision=0)
        batch_end = gr.Number(label="End Frame Index", precision=0)
        batch_label = gr.Radio(
            choices=list(LABEL_SHORT.values()),
            label="Label",
        )
        batch_btn = gr.Button("Apply Batch", variant="secondary")

    # ── Events ──
    traj_dropdown.change(
        on_select_trajectory,
        inputs=[traj_dropdown],
        outputs=[clean_img, ref_img, summary_box, frame_slider, frame_slider, current_label],
    ).then(on_refresh_timeline, outputs=[timeline_img])

    frame_slider.change(
        on_frame_change,
        inputs=[frame_slider],
        outputs=[clean_img, ref_img, current_label],
    )

    new_label.input(
        on_label_change,
        inputs=[frame_slider, new_label],
        outputs=[clean_img, ref_img, change_status],
    ).then(on_refresh_timeline, outputs=[timeline_img])

    prev_btn.click(on_prev, inputs=[frame_slider], outputs=[frame_slider])
    next_btn.click(on_next, inputs=[frame_slider], outputs=[frame_slider])

    save_btn.click(on_save, outputs=[summary_box])

    batch_btn.click(
        on_batch_label,
        inputs=[batch_start, batch_end, batch_label],
        outputs=[change_status],
    ).then(on_refresh_timeline, outputs=[timeline_img])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())
