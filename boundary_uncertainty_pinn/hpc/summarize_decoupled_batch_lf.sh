#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
STAMP=${1:?Usage: summarize_decoupled_batch_lf.sh STAMP [GROUP]}
GROUP=${2:-00.Smoke}
NODES=(10.6.234.22 10.6.234.23 10.6.234.24 10.6.234.25 10.6.234.27 10.6.234.28 10.6.234.29 10.6.234.30)
CASES=(A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15)

echo "===== summary time ====="
date "+%F %T"

echo "===== live processes on compute nodes ====="
for NODE in "${NODES[@]}"; do
  echo "--- $NODE ---"
  ssh -o BatchMode=yes -o ConnectTimeout=6 "wxtian@$NODE" "
    hostname
    ps -u \"\${USER:-wxtian}\" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= \
      | grep train_decoupled_pfnn.py \
      | grep '$PROJECT/exp/$GROUP/' \
      | grep -v grep || true
  " || true
done

echo "===== run directories ====="
ROOT="$PROJECT/exp/$GROUP"
for CASE in "${CASES[@]}"; do
  RUN_DIR="$ROOT/$CASE"
  LOG="$RUN_DIR/txt/train.log"
  DRIVER="$RUN_DIR/txt/driver.log"
  RUNTIME="$RUN_DIR/json/runtime_status.json"
  METRICS="$RUN_DIR/metrics/metrics.json"
  if [ ! -d "$RUN_DIR" ]; then
    echo "$CASE status=missing_run_dir run_dir=$RUN_DIR"
    continue
  fi
  if [ -f "$DRIVER" ] && ! grep -q "run_stamp=$STAMP" "$DRIVER"; then
    echo "$CASE status=stamp_mismatch run_dir=$RUN_DIR"
    continue
  fi
  STATUS=running_or_failed
  if [ -f "$RUNTIME" ]; then
    STATUS=completed
  fi
  if [ ! -f "$LOG" ]; then
    echo "$CASE status=missing_log run_dir=$RUN_DIR"
    continue
  fi
  LAST_STEP=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]' "$LOG" | tail -1 | awk '{print $1}' || true)
  STEP_LINE=$(grep -E 'STEP[[:space:]][0-9]+' "$LOG" | tail -1 || true)
  LAST_STEP=${LAST_STEP:-none}
  LOG_MTIME=$(stat -c '%y' "$LOG" | cut -d. -f1)
  ERROR_LINE=$(grep -E 'Traceback|ImportError|RuntimeError|ValueError|ERROR|Error' "$LOG" | tail -1 || true)
  PREFIX="$CASE status=$STATUS last_table_step=$LAST_STEP latest_step_line=\"${STEP_LINE}\" mtime=$LOG_MTIME"
  if [ -f "$METRICS" ]; then
    METRIC_LINE=$(python - "$METRICS" <<'PY'
import json
import sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)
rel = m.get("relative_l2", {})
print(
    " metrics="
    + " ".join(
        f"{k}={float(rel[k]):.4g}"
        for k in ("ux", "uy", "K", "mu", "E", "nu")
        if k in rel
    )
    + f" A0={m.get('final_A_or_beta0')}"
    + f" reaction={m.get('reaction_rel_error')}"
)
PY
)
    echo "$PREFIX $METRIC_LINE"
  else
    echo "$PREFIX metrics=not_written"
  fi
  if [ -n "$ERROR_LINE" ]; then
    echo "$CASE error=$ERROR_LINE"
  fi
done
