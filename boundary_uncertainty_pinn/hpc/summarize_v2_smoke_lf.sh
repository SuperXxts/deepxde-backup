#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:?Usage: summarize_v2_smoke_lf.sh STAMP}
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
ROOT="$PROJECT/exp/00.Smoke"

echo "===== V2 SMOKE SUMMARY ====="
date "+check_time=%F %T"
echo "node=$(hostname)"
echo "stamp=$STAMP"
echo "===== live process ====="
ps -u "${USER:-wxtian}" -f | grep train_deepxde_pfnn_v2.py | grep "$STAMP" | grep -v grep || true
echo "===== screen ====="
screen -ls | grep "bupinn_v2_smoke_$STAMP" || true

for CASE in full_boundary_oracle correct_top_amp_only wrong_fixed_A learnable_A learnable_A_reaction learnable_A_anchor_reaction; do
  RUN_DIR="$ROOT/v2_single_smoke_${CASE}_${STAMP}"
  LOG="$RUN_DIR/txt/train.log"
  METRICS="$RUN_DIR/metrics/metrics.json"
  echo "===== CASE $CASE ====="
  if [ ! -d "$RUN_DIR" ]; then
    echo "missing_run_dir"
    continue
  fi
  if [ -f "$LOG" ]; then
    stat -c "log_mtime=%y size=%s" "$LOG"
    grep -E "STEP [0-9]+|RUN_COMPLETE|Traceback|Error|Exception" "$LOG" | tail -20 || true
  else
    echo "missing_train_log"
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
    f"A={m.get('final_A'):.6f} "
    f"K={r.get('K', float('nan')):.6f} "
    f"mu={r.get('mu', float('nan')):.6f} "
    f"E={r.get('E', float('nan')):.6f} "
    f"reaction={m.get('reaction_rel_error'):.6f} "
    f"loss={m.get('final_train_loss_sum'):.6g}"
)
PY
    echo "artifact_counts:"
    find "$RUN_DIR" -maxdepth 2 -type f | sed "s#^$RUN_DIR/##" | cut -d/ -f1 | sort | uniq -c
  else
    echo "missing_metrics"
  fi
done
