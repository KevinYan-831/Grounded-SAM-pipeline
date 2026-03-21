#!/usr/bin/env bash
set -euo pipefail

# Root containing extracted trajectory frame dirs created by:
#   prepare_keyhole_labelme_workspace.py --data-root /home/jixin/xray-enhanced --output-root /home/jixin/xray-enhanced-frames
DATA_ROOT="/home/jixin/xray-enhanced-frames"

# Output root for labeling results
OUTPUT_ROOT="/home/jixin/labeling_trajectories_manual"

EXTRA_ARGS=("$@")

python run_labeling_batch.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --exts png \
  --manual-keyhole-from-labelme \
  --skip-existing \
  "${EXTRA_ARGS[@]}"
