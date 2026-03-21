#!/usr/bin/env python3
"""
Prepare a LabelMe workspace for manual keyhole annotation.

This script converts each multi-page TIFF trajectory into a directory of
frame images so LabelMe can be used to mark:
  - the first frame where keyhole appears (point)
  - the last frame where keyhole appears (point)

Expected annotation labels in LabelMe:
  - keyhole_start  (preferred)
  - keyhole_end    (preferred)

Fallback is also supported in labeling_pipeline_manual_keyhole.py:
  - keyhole (on exactly two frames: earliest=start, latest=end)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tifffile


DEFAULT_EXTS = (".tif", ".tiff")
DEFAULT_EXTS_STR = ",".join(e.lstrip(".") for e in DEFAULT_EXTS)


def _parse_exts(exts_arg: str) -> Tuple[str, ...]:
    if not exts_arg:
        return DEFAULT_EXTS
    parts = [p.strip().lower() for p in exts_arg.split(",") if p.strip()]
    if not parts:
        return DEFAULT_EXTS
    return tuple(f".{p}" if not p.startswith(".") else p for p in parts)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 for robust PNG/JPG writing."""
    if img.dtype == np.uint8:
        return img

    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    if np.issubdtype(img.dtype, np.integer):
        min_val = int(img.min())
        max_val = int(img.max())
    else:
        min_val = float(img.min())
        max_val = float(img.max())

    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)

    scaled = (img.astype(np.float32) - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return scaled


def _normalise_shape(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is HxW or HxWxC with C in {1,3,4} for OpenCV writing.
    """
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # If channels-first, move to channels-last.
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] in (1, 3, 4):
            return img

    raise ValueError(f"Unsupported TIFF page shape: {img.shape}")


def find_trajectory_files(data_root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for dirpath, _dirnames, filenames in os.walk(data_root):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(Path(dirpath) / name)
    return sorted(files)


def write_instructions(output_root: Path) -> None:
    content = """# Keyhole LabelMe Instructions

For each trajectory directory in this workspace:

1. Open LabelMe on that trajectory directory:
   labelme /path/to/trajectory_dir
2. Find the first frame where keyhole appears.
3. Create one point labeled `keyhole_start`.
4. Find the last frame where keyhole appears.
5. Create one point labeled `keyhole_end`.
6. Save JSON for both frames.

Notes:
- Do not annotate all frames; only start/end keyhole frames are needed.
- If you prefer, label both points as `keyhole` on two different frames.
  The pipeline will use earliest as start and latest as end.
"""
    (output_root / "KEYHOLE_LABELME_INSTRUCTIONS.md").write_text(content)


def extract_one_trajectory(
    tif_path: Path,
    data_root: Path,
    output_root: Path,
    image_ext: str,
    skip_existing: bool,
) -> Dict:
    rel_parent = tif_path.parent.relative_to(data_root)
    trajectory_name = tif_path.stem
    out_dir = output_root / rel_parent / trajectory_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(out_dir.glob(f"*.{image_ext}"))
    if skip_existing and existing_frames:
        return {
            "source_tif": str(tif_path),
            "trajectory_id": trajectory_name,
            "frames_dir": str(out_dir),
            "num_frames": len(existing_frames),
            "image_ext": image_ext,
            "skipped_existing": True,
        }

    with tifffile.TiffFile(str(tif_path)) as tf:
        pages = tf.pages
        num_pages = len(pages)
        height, width = pages[0].shape[:2]

        for i, page in enumerate(pages):
            arr = page.asarray()
            arr = _normalise_shape(arr)
            arr = _to_uint8(arr)

            out_path = out_dir / f"{i:05d}.{image_ext}"
            ok = cv2.imwrite(str(out_path), arr)
            if not ok:
                raise RuntimeError(f"Failed to write frame: {out_path}")

    return {
        "source_tif": str(tif_path),
        "trajectory_id": trajectory_name,
        "frames_dir": str(out_dir),
        "num_frames": num_pages,
        "image_ext": image_ext,
        "image_size": {"width": int(width), "height": int(height)},
        "skipped_existing": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare LabelMe workspace for keyhole start/end point annotation"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing trajectory TIFF stacks",
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Workspace directory for extracted trajectory frames",
    )
    parser.add_argument(
        "--exts", default=DEFAULT_EXTS_STR,
        help=f"Comma-separated input extensions (default: {DEFAULT_EXTS_STR})",
    )
    parser.add_argument(
        "--image-ext", default="png", choices=["png", "jpg", "jpeg"],
        help="Output frame image extension (default: png)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip trajectories that already have extracted frames",
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=0,
        help="Optional limit for quick testing (0 = all trajectories)",
    )
    parser.add_argument(
        "--manifest-name", default="keyhole_labelme_manifest.json",
        help="Manifest filename written under output-root",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    exts = _parse_exts(args.exts)
    image_ext = args.image_ext.lower().lstrip(".")
    output_root.mkdir(parents=True, exist_ok=True)

    tifs = find_trajectory_files(data_root, exts)
    if not tifs:
        print(f"No trajectory files found under {data_root} for extensions {exts}")
        return 1

    if args.max_trajectories > 0:
        tifs = tifs[: args.max_trajectories]

    print(f"Found {len(tifs)} trajectory TIFF files")
    print(f"Output workspace: {output_root}")

    trajectories = []
    for idx, tif_path in enumerate(tifs, start=1):
        print(f"[{idx:>3}/{len(tifs)}] {tif_path}")
        info = extract_one_trajectory(
            tif_path=tif_path,
            data_root=data_root,
            output_root=output_root,
            image_ext=image_ext,
            skip_existing=args.skip_existing,
        )
        trajectories.append(info)

    manifest = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "num_trajectories": len(trajectories),
        "annotation_labels": {
            "start": "keyhole_start",
            "end": "keyhole_end",
            "fallback": "keyhole",
        },
        "trajectories": trajectories,
    }
    manifest_path = output_root / args.manifest_name
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    write_instructions(output_root)

    print(f"\nManifest written to: {manifest_path}")
    print(
        "Next step: annotate each trajectory folder with LabelMe and save start/end points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
