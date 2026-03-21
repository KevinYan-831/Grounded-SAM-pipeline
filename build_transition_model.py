#!/usr/bin/env python3
"""
Aggregate labeling results across trajectories into a transition probability model.

Reads all labeling_results.json files under an output root and produces:
  1. Aggregated transition count matrix
  2. Transition probability matrix (row-normalized)
  3. Steady-state distribution (eigenvector)
  4. Per-trajectory summary table (CSV)

Usage:
    python build_transition_model.py \
        --results-root /home/jixin/labeling_trajectories_manual \
        --output transition_model.json \
        --csv transition_summary.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

LABEL_NAMES = [
    "Normal Process",
    "Unstable Process without Pore Generation",
    "Permanent Pore Generation",
    "Transient Pore Generation",
]
NUM_LABELS = len(LABEL_NAMES)


def find_result_files(results_root: str):
    """Find all labeling_results.json files recursively."""
    found = []
    for dirpath, _, filenames in os.walk(results_root):
        for fname in filenames:
            if fname == "labeling_results.json":
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def compute_transition_counts(label_seq):
    """Compute transition count matrix from a label sequence."""
    counts = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    for i in range(len(label_seq) - 1):
        src, dst = label_seq[i], label_seq[i + 1]
        if 0 <= src < NUM_LABELS and 0 <= dst < NUM_LABELS:
            counts[src, dst] += 1
    return counts


def normalize_rows(counts):
    """Row-normalize a count matrix to get probabilities."""
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    return counts.astype(float) / row_sums


def steady_state(prob_matrix):
    """Compute steady-state distribution via eigenvalue decomposition."""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(prob_matrix.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        ss = np.real(eigenvectors[:, idx])
        ss = ss / ss.sum()  # normalize
        if np.any(ss < 0):
            ss = np.abs(ss)
            ss = ss / ss.sum()
        return ss
    except Exception:
        return np.zeros(NUM_LABELS)


def main():
    parser = argparse.ArgumentParser(
        description="Build transition probability model from labeling results"
    )
    parser.add_argument(
        "--results-root", required=True,
        help="Root directory containing per-trajectory labeling_results.json files",
    )
    parser.add_argument(
        "--output", default="transition_model.json",
        help="Output JSON file for the transition model (default: transition_model.json)",
    )
    parser.add_argument(
        "--csv", default="transition_summary.csv",
        help="Output CSV with per-trajectory summary (default: transition_summary.csv)",
    )
    parser.add_argument(
        "--use-analyzed-only", action="store_true", default=True,
        help="Use only the analyzed frame range (keyhole present) for transitions (default: True)",
    )
    args = parser.parse_args()

    result_files = find_result_files(args.results_root)
    if not result_files:
        print(f"No labeling_results.json files found under {args.results_root}")
        return 1

    print(f"Found {len(result_files)} trajectory results")

    global_counts = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    per_trajectory = []

    for rpath in result_files:
        rel = os.path.relpath(rpath, args.results_root)
        traj_id = os.path.dirname(rel)

        with open(rpath) as f:
            data = json.load(f)

        # Use pre-computed transition counts if available
        if "transition_counts" in data:
            tc = np.array(data["transition_counts"], dtype=int)
            global_counts += tc
        else:
            # Fall back to computing from sequence
            seq_key = "analyzed_label_sequence" if args.use_analyzed_only else "label_sequence"
            seq = data.get(seq_key)
            if seq is None:
                # Legacy: extract from frame_labels
                seq = [fl["label_id"] for fl in data.get("frame_labels", [])]
            tc = compute_transition_counts(seq)
            global_counts += tc

        # Per-trajectory summary
        summary = data.get("summary", {})
        total = data.get("total_frames", 0)
        analysis_range = data.get("analysis_frame_range", [0, total - 1])

        traj_info = {
            "trajectory_id": traj_id,
            "total_frames": total,
            "analyzed_frames": analysis_range[1] - analysis_range[0] + 1 if analysis_range else total,
            "transition_counts": tc.tolist(),
        }
        for lid, name in enumerate(LABEL_NAMES):
            traj_info[f"count_{name}"] = summary.get(name, 0)

        per_trajectory.append(traj_info)

    # Global model
    prob_matrix = normalize_rows(global_counts)
    ss_dist = steady_state(prob_matrix)

    print("\n" + "=" * 70)
    print("TRANSITION PROBABILITY MATRIX")
    print("=" * 70)

    # Header
    short_names = ["Normal", "Unstable", "Permanent", "Transient"]
    header = f"{'From \\ To':>14s}" + "".join(f"{n:>12s}" for n in short_names)
    print(header)
    print("-" * len(header))

    for i in range(NUM_LABELS):
        row = f"{short_names[i]:>14s}"
        for j in range(NUM_LABELS):
            row += f"{prob_matrix[i, j]:12.4f}"
        row += f"  (n={global_counts[i].sum():>5d})"
        print(row)

    print(f"\n{'Steady-state':>14s}" + "".join(f"{v:12.4f}" for v in ss_dist))
    print(f"\nTotal transitions: {global_counts.sum()}")
    print(f"Trajectories:      {len(per_trajectory)}")

    # Save model JSON
    model_output = {
        "num_trajectories": len(per_trajectory),
        "label_names": LABEL_NAMES,
        "transition_count_matrix": global_counts.tolist(),
        "transition_probability_matrix": prob_matrix.tolist(),
        "steady_state_distribution": ss_dist.tolist(),
        "per_trajectory": per_trajectory,
    }

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.results_root, output_path)
    with open(output_path, "w") as f:
        json.dump(model_output, f, indent=2)
    print(f"\nTransition model saved to {output_path}")

    # Save CSV
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.results_root, csv_path)

    fieldnames = ["trajectory_id", "total_frames", "analyzed_frames"]
    fieldnames += [f"count_{name}" for name in LABEL_NAMES]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in per_trajectory:
            writer.writerow(t)
    print(f"Per-trajectory CSV saved to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
