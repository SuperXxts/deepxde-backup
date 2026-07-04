#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:?Usage: check_v2_case_live_lf.sh STAMP CASE [WAIT_SECONDS]}
CASE=${2:?Usage: check_v2_case_live_lf.sh STAMP CASE [WAIT_SECONDS]}
WAIT_SECONDS=${3:-30}

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
ROOT="$PROJECT/exp/01.AnalyticalBenchmark"
RUN_DIR="$ROOT/v2_mms_single_obs225_seed42_${CASE}_${STAMP}"
LOG="$RUN_DIR/txt/train.log"
SERIAL_SCREEN="bupinn_v2_analytical6_150k_${STAMP}"
ONECASE_SCREEN="bupinn_v2_150k_${CASE}_${STAMP}"

echo "===== CASE LIVE CHECK ====="
echo "node=$(hostname)"
date "+check_time=%F %T"
echo "case=$CASE"
echo "stamp=$STAMP"
echo "run_dir=$RUN_DIR"
echo "wait_seconds=$WAIT_SECONDS"

echo "===== PROCESS ====="
ps -u "${USER:-wxtian}" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= \
  | grep train_deepxde_pfnn_v2.py \
  | grep "$RUN_DIR" \
  | grep -v grep || true

echo "===== SCREEN ====="
screen -ls | grep -E "(${SERIAL_SCREEN}|${ONECASE_SCREEN})" || true

echo "===== LOG BEFORE ====="
if [ ! -f "$LOG" ]; then
  echo "missing_log=$LOG"
  exit 0
fi
stat -c "mtime_before=%y size_before=%s" "$LOG"
STEP_BEFORE=$(grep -E "STEP [0-9]+" "$LOG" | tail -1 || true)
echo "step_before=$STEP_BEFORE"

sleep "$WAIT_SECONDS"

echo "===== LOG AFTER ====="
stat -c "mtime_after=%y size_after=%s" "$LOG"
STEP_AFTER=$(grep -E "STEP [0-9]+" "$LOG" | tail -1 || true)
echo "step_after=$STEP_AFTER"

python - "$STEP_BEFORE" "$STEP_AFTER" <<'PY'
import re
import sys

def parse_step(line):
    match = re.search(r"STEP\s+([0-9]+)", line or "")
    return int(match.group(1)) if match else None

before = parse_step(sys.argv[1])
after = parse_step(sys.argv[2])
print(f"step_before_num={before}")
print(f"step_after_num={after}")
if before is not None and after is not None:
    print(f"step_delta={after - before}")
    print(f"step_growing={after > before}")
else:
    print("step_delta=unknown")
    print("step_growing=unknown")
PY
