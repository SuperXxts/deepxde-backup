#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:?Usage: check_v2_dcu_status_lf.sh STAMP}
CASE=${2:-full_boundary_oracle}
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_DIR="$PROJECT/exp/01.AnalyticalBenchmark/v2_mms_single_obs225_seed42_${CASE}_${STAMP}"
SCREEN_NAME="bupinn_v2_analytical6_150k_${STAMP}"
LOG="$RUN_DIR/txt/train.log"

echo "===== NODE ====="
hostname
date "+check_time=%F %T"
echo "run_dir=$RUN_DIR"

echo "===== PROCESS ====="
PIDS=$(ps -u "${USER:-wxtian}" -o pid=,ppid=,pcpu=,pmem=,etime=,cmd= | grep train_deepxde_pfnn_v2.py | grep "$STAMP" | grep -v grep | awk '{print $1}' || true)
if [ -z "$PIDS" ]; then
  echo "no_live_training_process"
else
  for PID in $PIDS; do
    ps -p "$PID" -o pid,ppid,pcpu,pmem,etime,stat,cmd
    echo "--- env $PID ---"
    tr '\0' '\n' < "/proc/$PID/environ" | grep -E 'CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|DDE_BACKEND|CONDA_DEFAULT_ENV|PATH|LD_LIBRARY_PATH' || true
  done
fi

echo "===== SCREEN ====="
screen -ls | grep "$SCREEN_NAME" || true

echo "===== LOG STATUS ====="
if [ -f "$LOG" ]; then
  stat -c "mtime=%y size=%s" "$LOG"
  grep -E "^[[:space:]]*[0-9]+[[:space:]]|STEP [0-9]+|RUN_COMPLETE|Traceback|Error|Exception" "$LOG" | tail -30 || true
else
  echo "missing_log=$LOG"
fi

echo "===== PYTORCH DEVICE FROM LOG/CONFIG ====="
CONFIG="$RUN_DIR/json/config.json"
if [ -f "$CONFIG" ]; then
  python - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = json.load(f)
for key in ["backend", "torch_cuda_available", "torch_device_count", "network", "case", "material_mode"]:
    print(f"{key}={c.get(key)}")
PY
else
  echo "missing_config=$CONFIG"
fi

echo "===== DCU/GPU MONITOR ====="
for cmd in hy-smi rocm-smi nvidia-smi; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "--- $cmd ---"
    "$cmd" 2>&1 | head -120 || true
  else
    echo "$cmd not found"
  fi
done
