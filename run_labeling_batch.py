#!/usr/bin/env python3
"""
Batch runner for labeling pipelines.

Scans a data root for trajectory directories containing frame files,
then runs a labeling pipeline for each trajectory with per-trajectory
outputs.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Tuple

DEFAULT_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
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


def resolve_pipeline_script(pipeline_script: str, manual_keyhole_from_labelme: bool) -> str:
    chosen = pipeline_script
    if manual_keyhole_from_labelme and pipeline_script == "labeling_pipeline.py":
        chosen = "labeling_pipeline_manual_keyhole.py"

    if os.path.isabs(chosen):
        return chosen
    return os.path.join(os.path.dirname(__file__), chosen)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a labeling pipeline over multiple trajectories"
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
        "--pipeline-script", default="labeling_pipeline.py",
        help="Pipeline script filename/path (default: labeling_pipeline.py)",
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

    # Manual-keyhole-specific flags (forwarded when enabled)
    parser.add_argument(
        "--manual-keyhole-from-labelme", action="store_true",
        help=(
            "Use LabelMe point annotations (start/end keyhole) per trajectory. "
            "If --pipeline-script is left at default, this switches to "
            "labeling_pipeline_manual_keyhole.py automatically."
        ),
    )
    parser.add_argument(
        "--labelme-dir-name", default="",
        help=(
            "Optional subdirectory under each trajectory containing LabelMe JSONs. "
            "Default: empty (JSONs are in the trajectory frame directory itself)."
        ),
    )
    parser.add_argument(
        "--manual-start-label", default="keyhole_start",
        help="LabelMe point label for keyhole start (default: keyhole_start)",
    )
    parser.add_argument(
        "--manual-end-label", default="keyhole_end",
        help="LabelMe point label for keyhole end (default: keyhole_end)",
    )
    parser.add_argument(
        "--manual-fallback-label", default="keyhole",
        help="Fallback LabelMe point label (default: keyhole)",
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)
    exts = _parse_exts(args.exts)

    frame_dirs = find_frame_dirs(data_root, exts)
    if not frame_dirs:
        print(f"No frame directories found under {data_root} for extensions {exts}")
        return 1

    script_path = resolve_pipeline_script(
        pipeline_script=args.pipeline_script,
        manual_keyhole_from_labelme=args.manual_keyhole_from_labelme,
    )
    if not os.path.exists(script_path):
        print(f"Pipeline script not found at {script_path}")
        return 1

    print(f"Found {len(frame_dirs)} trajectories to process")
    print(f"Using pipeline script: {script_path}")

    skipped_unannotated = []

    for frame_dir in frame_dirs:
        rel = os.path.relpath(frame_dir, data_root)
        out_dir = os.path.join(output_root, rel)
        output_json = os.path.join(out_dir, "labeling_results.json")
        output_frames_dir = os.path.join(out_dir, "frames")
        detection_cache = os.path.join(out_dir, "raw_detections.json")

        if args.skip_existing and os.path.exists(output_json):
            print(f"Skipping (exists): {output_json}")
            continue

        # Skip trajectories without keyhole annotations in manual mode
        if args.manual_keyhole_from_labelme:
            labelme_search_dir = frame_dir
            if args.labelme_dir_name:
                labelme_search_dir = os.path.join(frame_dir, args.labelme_dir_name)
            has_keyhole_ann = False
            for fname in os.listdir(labelme_search_dir):
                if not fname.lower().endswith(".json"):
                    continue
                jpath = os.path.join(labelme_search_dir, fname)
                try:
                    with open(jpath) as jf:
                        data = json.load(jf)
                    for shape in data.get("shapes", []):
                        lbl = str(shape.get("label", "")).strip().lower()
                        if lbl in (
                            args.manual_start_label.lower(),
                            args.manual_end_label.lower(),
                            args.manual_fallback_label.lower(),
                        ):
                            has_keyhole_ann = True
                            break
                except Exception:
                    pass
                if has_keyhole_ann:
                    break
            if not has_keyhole_ann:
                skipped_unannotated.append(rel)
                print(f"Skipping (no keyhole annotation): {rel}")
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

        if args.manual_keyhole_from_labelme:
            cmd.extend([
                "--manual-start-label", args.manual_start_label,
                "--manual-end-label", args.manual_end_label,
                "--manual-fallback-label", args.manual_fallback_label,
            ])
            if args.labelme_dir_name:
                labelme_dir = os.path.join(frame_dir, args.labelme_dir_name)
                cmd.extend(["--labelme-dir", labelme_dir])

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

    if skipped_unannotated:
        print(f"\nSkipped {len(skipped_unannotated)} unannotated trajectories:")
        for s in skipped_unannotated:
            print(f"  - {s}")

    print("\nAll trajectories processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
