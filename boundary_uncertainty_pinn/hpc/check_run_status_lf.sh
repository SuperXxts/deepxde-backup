#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?Usage: check_run_status_lf.sh RUN_DIR SCREEN_NAME PROCESS_PATTERN}
SCREEN_NAME=${2:-}
PROCESS_PATTERN=${3:-train_deepxde_pfnn.py}
LOG_FILE="$RUN_DIR/txt/train.log"
DRIVER_LOG="$RUN_DIR/txt/driver.log"
METRICS_FILE="$RUN_DIR/metrics/metrics.json"

echo "===== STATUS CHECK ====="
echo "node=$(hostname)"
date "+check_time=%F %T"
echo "run_dir=$RUN_DIR"

echo "===== PROCESS ====="
ps -u "${USER:-wxtian}" -f | grep "$PROCESS_PATTERN" | grep "$RUN_DIR" | grep -v grep || true

echo "===== SCREEN ====="
if [ -n "$SCREEN_NAME" ]; then
  screen -ls | grep "$SCREEN_NAME" || true
else
  screen -ls || true
fi

echo "===== LOG STAT ====="
if [ -f "$LOG_FILE" ]; then
  stat -c "train_log_mtime=%y size=%s" "$LOG_FILE"
else
  echo "train_log_missing=$LOG_FILE"
fi
if [ -f "$DRIVER_LOG" ]; then
  stat -c "driver_log_mtime=%y size=%s" "$DRIVER_LOG"
else
  echo "driver_log_missing=$DRIVER_LOG"
fi

echo "===== STEP LINES ====="
if [ -f "$LOG_FILE" ]; then
  grep -E "^[[:space:]]*[0-9]+[[:space:]]|STEP [0-9]+|RUN_COMPLETE|Training model|Compiling model|Best model" "$LOG_FILE" | tail -40 || true
fi

echo "===== LOG TAIL ====="
if [ -f "$LOG_FILE" ]; then
  tail -80 "$LOG_FILE"
fi

echo "===== METRICS ====="
if [ -f "$METRICS_FILE" ]; then
  cat "$METRICS_FILE"
else
  echo "metrics_missing=$METRICS_FILE"
fi
