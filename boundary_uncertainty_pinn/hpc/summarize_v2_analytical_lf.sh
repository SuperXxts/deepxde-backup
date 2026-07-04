#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:?Usage: summarize_v2_analytical_lf.sh STAMP}
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
ROOT="$PROJECT/exp/01.AnalyticalBenchmark"
SCREEN_NAME="bupinn_v2_analytical6_150k_$STAMP"

echo "===== V2 ANALYTICAL SUMMARY ====="
echo "node=$(hostname)"
date "+check_time=%F %T"
echo "stamp=$STAMP"

echo "===== LIVE PROCESS ====="
ps -u "${USER:-wxtian}" -o pid=,ppid=,pcpu=,pmem=,etime=,cmd= | grep train_deepxde_pfnn_v2.py | grep "$STAMP" | grep -v grep || true

echo "===== SCREEN ====="
screen -ls | grep "$SCREEN_NAME" || true

for CASE in full_boundary_oracle correct_top_amp_only wrong_fixed_A learnable_A learnable_A_reaction learnable_A_anchor_reaction; do
  RUN_DIR="$ROOT/v2_mms_single_obs225_seed42_${CASE}_${STAMP}"
  LOG="$RUN_DIR/txt/train.log"
  METRICS="$RUN_DIR/metrics/metrics.json"
  DRIVER="$RUN_DIR/txt/driver.log"
  echo "===== CASE $CASE ====="
  echo "run_dir=$RUN_DIR"
  if [ ! -d "$RUN_DIR" ]; then
    echo "status=missing_run_dir"
    continue
  fi
  if [ -f "$DRIVER" ]; then
    stat -c "driver_mtime=%y size=%s" "$DRIVER"
    tail -8 "$DRIVER"
  else
    echo "driver_missing"
  fi
  if [ -f "$LOG" ]; then
    stat -c "train_mtime=%y size=%s" "$LOG"
    grep -E "^[[:space:]]*[0-9]+[[:space:]]|STEP [0-9]+|RUN_COMPLETE|Traceback|Error|Exception" "$LOG" | tail -25 || true
  else
    echo "train_log_missing"
  fi
  if [ -f "$METRICS" ]; then
    python - "$METRICS" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    m = json.load(f)
r = m.get("relative_l2", {})
print(
    "metrics "
    f"case={m.get('case')} "
    f"iters={m.get('iterations')} "
    f"elapsed={m.get('elapsed_seconds'):.1f} "
    f"A={m.get('final_A'):.8f} "
    f"K={r.get('K', float('nan')):.6f} "
    f"mu={r.get('mu', float('nan')):.6f} "
    f"E={r.get('E', float('nan')):.6f} "
    f"ux={r.get('ux', float('nan')):.6f} "
    f"uy={r.get('uy', float('nan')):.6f} "
    f"reaction={m.get('reaction_rel_error'):.6f} "
    f"loss={m.get('final_train_loss_sum'):.6g} "
    f"best_step={m.get('best_step')}"
)
PY
  else
    echo "metrics_missing"
  fi
done
