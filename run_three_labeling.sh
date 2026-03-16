#!/usr/bin/env bash
set -euo pipefail

#output directory
OUTPUT_ROOT="/home/jixin/labeling_trajectories"

# Update these three inputs and outputs.
DATA_ROOT_1="/home/jixin/x-ray-data/xray_001_065_Ti64_plate"


DATA_ROOT_2="/home/jixin/x-ray-data/xray_066_134_Ti64_powder"


DATA_ROOT_3="/home/jixin/x-ray-data/xray_135_150_Ti64_others"


EXTRA_ARGS=("$@")

run_one() {
  local data_root="$1"
  local output_root="$2"
  echo ""
  echo "============================================================"
  echo "Running labeling batch"
  echo "  data_root:   ${data_root}"
  echo "  output_root: ${output_root}"
  python run_labeling_batch.py \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --skip-existing \
    "${EXTRA_ARGS[@]}"
}

run_one "${DATA_ROOT_1}" "${OUTPUT_ROOT}"
run_one "${DATA_ROOT_2}" "${OUTPUT_ROOT}"
run_one "${DATA_ROOT_3}" "${OUTPUT_ROOT}"
