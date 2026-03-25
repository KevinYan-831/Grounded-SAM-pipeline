#!/usr/bin/env python3
"""
Upload labeled dataset to HuggingFace.
Uses clean frames from xray-enhanced-frames + labeling results JSONs.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RESULTS_ROOT = "/home/jixin/labeling_trajectories_manual"
CLEAN_ROOT = "/home/jixin/xray-enhanced-frames"
REPO_ID = "kevin831/xray-welding-labeled-frames"

# Top-level files to include from results root
TOP_FILES = [
    "README.md",
    "transition_model.json",
    "transition_summary.csv",
    "transition_model_visualization.png",
]


def build_staging_dir(staging_dir):
    """Build staging directory with symlinks to avoid data duplication."""
    staging = Path(staging_dir)

    # Copy top-level files
    for fname in TOP_FILES:
        src = Path(RESULTS_ROOT) / fname
        if src.exists():
            os.symlink(src, staging / fname)
            print(f"  + {fname}")

    # Process each trajectory
    traj_count = 0
    frame_count = 0
    for result_path in sorted(Path(RESULTS_ROOT).rglob("labeling_results.json")):
        traj_rel = result_path.parent.relative_to(RESULTS_ROOT)
        traj_staging = staging / traj_rel

        # Load labeling results to get frame list
        with open(result_path) as f:
            data = json.load(f)

        frame_labels = data.get("frame_labels", [])
        if not frame_labels:
            continue

        # Create trajectory directory
        traj_staging.mkdir(parents=True, exist_ok=True)

        # Symlink labeling_results.json
        os.symlink(result_path, traj_staging / "labeling_results.json")

        # Symlink clean frames (only for analyzed frames)
        clean_traj_dir = Path(CLEAN_ROOT) / traj_rel
        if not clean_traj_dir.exists():
            print(f"  WARNING: clean frames not found for {traj_rel}")
            continue

        frames_staging = traj_staging / "frames"
        frames_staging.mkdir(exist_ok=True)

        for fl in frame_labels:
            fidx = fl["frame_index"]
            # Try png then jpg
            for ext in ["png", "jpg"]:
                src = clean_traj_dir / f"{fidx:05d}.{ext}"
                if src.exists():
                    os.symlink(src, frames_staging / f"{fidx:05d}.{ext}")
                    frame_count += 1
                    break

        traj_count += 1

    print(f"\nStaged {traj_count} trajectories, {frame_count} clean frames")
    return traj_count, frame_count


def main():
    staging_dir = os.path.join(RESULTS_ROOT, ".upload_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    print("Building staging directory with clean frames...")
    traj_count, frame_count = build_staging_dir(staging_dir)

    print(f"\nUploading to {REPO_ID}...")
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        folder_path=staging_dir,
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Clean up staging dir
    shutil.rmtree(staging_dir)
    print("Done! Staging directory cleaned up.")


if __name__ == "__main__":
    main()
