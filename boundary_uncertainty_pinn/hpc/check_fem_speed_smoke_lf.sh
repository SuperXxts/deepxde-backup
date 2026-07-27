#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
GROUP=${1:-02.FEMThreeLoadSpeedSmoke}

echo "group=$GROUP"
date "+time=%F %T"

RUNS=(
  S0_B1_default
  S1_B1_no_dynamic_fig
  S2_B1_no_smooth
  S3_B6_no_dynamic_fig
  S4_B6_no_smooth
)

echo "== live processes =="
for node in comput1 comput2 comput3 comput4 comput5 comput6 comput7 comput8; do
  echo "-- $node --"
  ssh "$node" "ps -u '$USER' -o pid=,stat=,etime=,args= | grep '$PROJECT/exp/$GROUP/' | grep 'scripts/train_fem_multiload_gate.py' | grep -v grep || true"
done

echo "== run logs =="
for run in "${RUNS[@]}"; do
  dir="$PROJECT/exp/$GROUP/$run"
  log="$dir/txt/train.log"
  runtime="$dir/json/runtime_status.json"
  metrics="$dir/metrics/metrics.json"
  echo "-- $run --"
  if [ ! -d "$dir" ]; then
    echo "NO_DIR"
    continue
  fi
  if [ -f "$log" ]; then
    stat -c "log_mtime=%y" "$log"
    last_step=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]' "$log" | tail -n 1 | awk '{print $1}' || true)
    last_beta=$(grep '^STEP ' "$log" | tail -n 1 || true)
    train_took=$(grep "'train' took" "$log" | tail -n 1 || true)
    echo "last_step=${last_step:-NA}"
    if [ -n "$last_beta" ]; then
      echo "$last_beta"
    fi
    if [ -n "$train_took" ]; then
      echo "$train_took"
    fi
  else
    echo "NO_LOG"
  fi
  if [ -f "$runtime" ]; then
    tr -d '\n' < "$runtime"
    echo
  fi
  if [ -f "$metrics" ]; then
    python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print("elapsed_seconds=%s iterations=%s best_step=%s case=%s" % (d.get("elapsed_seconds"), d.get("iterations"), d.get("best_step"), d.get("case"))); print("relative_l2=%s" % d.get("relative_l2", {}))' "$metrics"
  fi
done
