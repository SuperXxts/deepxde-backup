#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: bash benchmark/scripts/launch_spatial_train_screen.sh <screen_name> <device_id> --method piminn --case single_inclusion ..."
  exit 1
fi

SCREEN_NAME=$1
DEVICE_ID=$2
shift 2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CMD="export CUDA_VISIBLE_DEVICES=$DEVICE_ID; bash $(printf '%q' "$SCRIPT_DIR/run_spatial_train.sh")"
for ARG in "$@"; do
  CMD+=" $(printf '%q' "$ARG")"
done
screen -S "$SCREEN_NAME" -dm bash -lc "$CMD"
screen -ls | grep "$SCREEN_NAME" || true
