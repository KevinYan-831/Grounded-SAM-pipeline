#!/usr/bin/env python3
"""
Batch runner for labeling_pipeline.py.

Scans a data root for trajectory directories containing frame files,
then runs the labeling pipeline for each trajectory with per-trajectory
outputs.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

DEFAULT_EXTS = (".tif", ".tiff")
DEFAULT_EXTS_STR = ",".join(e.lstrip(".") for e in DEFAULT_EXTS)


def _parse_exts(exts_arg: str) -> Tuple[str, ...]:
    if not exts_arg:
        return DEFAULT_EXTS
    parts = [p.strip().lower() for p in exts_arg.split(",") if p.strip()]
    if not parts:
        return DEFAULT_EXTS
    return tuple(f".{p}" if not p.startswith(".") else p for p in parts)


def find_frame_dirs(data_root: str, exts: Tuple[str, ...]) -> List[str]:
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(data_root):
        if any(f.lower().endswith(exts) for f in filenames):
            candidates.append(dirpath)

    # Keep only leaf-most directories (avoid double-processing parents)
    candidates_set = set(candidates)
    leaf_dirs = []
    for d in candidates:
        prefix = d + os.sep
        if any(other.startswith(prefix) for other in candidates_set if other != d):
            continue
        leaf_dirs.append(d)

    return sorted(leaf_dirs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run labeling_pipeline.py over multiple trajectories"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing trajectory folders",
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Root directory for per-trajectory outputs",
    )
    parser.add_argument(
        "--config", default="labeling_rules.yaml",
        help="Labeling rules YAML (default: labeling_rules.yaml)",
    )
    parser.add_argument(
        "--exts", default=DEFAULT_EXTS_STR,
        help=f"Comma-separated frame extensions (default: {DEFAULT_EXTS_STR})",
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Reuse cached detections if present",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip trajectories that already have labeling_results.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)
    exts = _parse_exts(args.exts)

    frame_dirs = find_frame_dirs(data_root, exts)
    if not frame_dirs:
        print(f"No frame directories found under {data_root} for extensions {exts}")
        return 1

    script_path = os.path.join(os.path.dirname(__file__), "labeling_pipeline.py")
    if not os.path.exists(script_path):
        print(f"labeling_pipeline.py not found at {script_path}")
        return 1

    print(f"Found {len(frame_dirs)} trajectories to process")

    for frame_dir in frame_dirs:
        rel = os.path.relpath(frame_dir, data_root)
        out_dir = os.path.join(output_root, rel)
        output_json = os.path.join(out_dir, "labeling_results.json")
        output_frames_dir = os.path.join(out_dir, "frames")
        detection_cache = os.path.join(out_dir, "raw_detections.json")

        if args.skip_existing and os.path.exists(output_json):
            print(f"Skipping (exists): {output_json}")
            continue

        cmd = [
            sys.executable,
            script_path,
            "--config", args.config,
            "--frames-dir", frame_dir,
            "--output", output_json,
            "--output-frames-dir", output_frames_dir,
            "--detection-cache", detection_cache,
            "--video-path", frame_dir,
            "--skip-extraction",
        ]
        if args.skip_detection:
            cmd.append("--skip-detection")

        print("\n" + "=" * 60)
        print(f"Trajectory: {frame_dir}")
        print("Command:", " ".join(cmd))

        if args.dry_run:
            continue

        os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: labeling failed for {frame_dir} (exit code {result.returncode})")
            return result.returncode

    print("\nAll trajectories processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
