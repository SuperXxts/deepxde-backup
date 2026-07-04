#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:?Usage: summarize_batch_lf.sh RUN_STAMP}
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
ROOT="$PROJECT/exp/01.AnalyticalBenchmark"

echo "===== BATCH SUMMARY ====="
echo "node=$(hostname)"
date "+check_time=%F %T"
echo "stamp=$STAMP"
echo "root=$ROOT"

for CASE in oracle_A wrong_fixed_A learnable_A learnable_A_anchor learnable_A_reaction full_boundary_aware; do
  RUN_DIR="$ROOT/mms_smooth_lens_obs225_A085_seed42_${CASE}_${STAMP}"
  LOG="$RUN_DIR/txt/train.log"
  DRIVER="$RUN_DIR/txt/driver.log"
  METRICS="$RUN_DIR/metrics/metrics.json"
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
    echo "last_steps:"
    grep -E "^[[:space:]]*[0-9]+[[:space:]]|STEP [0-9]+|RUN_COMPLETE|Traceback|Error|Exception" "$LOG" | tail -25 || true
  else
    echo "train_log_missing"
  fi
  if [ -f "$METRICS" ]; then
    echo "metrics_present=yes"
    python - "$METRICS" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)
rel = m.get("relative_l2", {})
print(
    "metrics "
    f"case={m.get('case')} "
    f"iters={m.get('iterations')} "
    f"elapsed={m.get('elapsed_seconds'):.3f} "
    f"A={m.get('final_A'):.8f} "
    f"K={rel.get('K', float('nan')):.6f} "
    f"mu={rel.get('mu', float('nan')):.6f} "
    f"E={rel.get('E', float('nan')):.6f} "
    f"ux={rel.get('ux', float('nan')):.6f} "
    f"uy={rel.get('uy', float('nan')):.6f} "
    f"reaction={m.get('reaction_rel_error'):.6f} "
    f"best_step={m.get('best_step')}"
)
PY
  else
    echo "metrics_present=no"
  fi
  echo "artifact_counts:"
  find "$RUN_DIR" -maxdepth 2 -type f | sed "s#^$RUN_DIR/##" | cut -d/ -f1 | sort | uniq -c
done
