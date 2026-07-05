#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
GROUP=${1:-02.FEMBenchmarkSmoke}
STAMP=${2:-}
NODES=(10.6.234.22 10.6.234.23 10.6.234.24 10.6.234.25 10.6.234.27 10.6.234.28 10.6.234.29 10.6.234.30)

echo "===== check time ====="
date "+%F %T"
echo "group=$GROUP"
echo "stamp=$STAMP"

echo "===== live training processes ====="
for NODE in "${NODES[@]}"; do
  echo "--- $NODE ---"
  ssh "$NODE" "hostname; ps -u \"\${USER:-wxtian}\" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= | grep train_fem_benchmark.py | grep '$PROJECT/exp/$GROUP/' | grep -v grep || true"
done

echo "===== run directories ====="
for CASE in B0 B1 B2 B3 B4 B5 B6 B7; do
  RUN_DIR="$PROJECT/exp/$GROUP/$CASE"
  if [ ! -d "$RUN_DIR" ]; then
    echo "$CASE status=missing run_dir=$RUN_DIR"
    continue
  fi
  DRIVER="$RUN_DIR/txt/driver.log"
  TRAIN="$RUN_DIR/txt/train.log"
  STATUS="present"
  if [ -f "$RUN_DIR/json/runtime_status.json" ]; then
    STATUS="completed"
  elif [ -f "$TRAIN" ]; then
    STATUS="has_train_log"
  fi
  if [ -n "$STAMP" ] && [ -f "$DRIVER" ] && ! grep -q "run_stamp=$STAMP" "$DRIVER"; then
    STATUS="stamp_mismatch"
  fi
  STEP=$(grep -E "^STEP[[:space:]]+[0-9]+" "$TRAIN" 2>/dev/null | tail -n 1 | awk '{print $2}' || true)
  LOSS_STEP=$(grep -E "^[[:space:]]*[0-9]+[[:space:]]+" "$TRAIN" 2>/dev/null | tail -n 1 | awk '{print $1}' || true)
  MTIME=$(stat -c "%y" "$TRAIN" 2>/dev/null | cut -d'.' -f1 || true)
  METRICS=""
  if [ -f "$RUN_DIR/metrics/metrics.json" ]; then
    METRICS=$(python - "$RUN_DIR/metrics/metrics.json" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p, encoding='utf-8'))
r=d.get("relative_l2", {})
print("K={:.4g} mu={:.4g} E={:.4g} nu={:.4g} reaction={:.4g}".format(
    r.get("K", float("nan")), r.get("mu", float("nan")),
    r.get("E", float("nan")), r.get("nu", float("nan")),
    d.get("reaction_rel_error", float("nan"))))
PY
)
  fi
  echo "$CASE status=$STATUS step=${STEP:-NA} loss_step=${LOSS_STEP:-NA} log_mtime=${MTIME:-NA} $METRICS"
done
