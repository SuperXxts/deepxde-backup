#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
GROUP=${1:-02.FEMOfficialPFNNSpeedSmoke}

echo "group=$GROUP"
date "+time=%F %T"

echo "== live processes =="
for node in comput1 comput2 comput3 comput4 comput5 comput6 comput7 comput8; do
  echo "-- $node --"
  ssh "$node" "ps -u '$USER' -o pid=,stat=,etime=,args= | grep '$PROJECT/exp/$GROUP/' | grep 'scripts/train_fem_official_pfnn_speed.py' | grep -v grep || true"
done

echo "== run logs =="
for run in P0 P1 P2; do
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
    speed_line=$(grep '^STEP ' "$log" | tail -n 1 || true)
    train_took=$(grep "'train' took" "$log" | tail -n 1 || true)
    err_line=$(grep -Ei 'traceback|runtimeerror|valueerror|killed|nan' "$log" | tail -n 1 || true)
    echo "last_step=${last_step:-NA}"
    if [ -n "$speed_line" ]; then
      echo "$speed_line"
    fi
    if [ -n "$train_took" ]; then
      echo "$train_took"
    fi
    if [ -n "$err_line" ]; then
      echo "ERR=$err_line"
    fi
  else
    echo "NO_LOG"
  fi
  if [ -f "$runtime" ]; then
    tr -d '\n' < "$runtime"
    echo
  fi
  if [ -f "$metrics" ]; then
    python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print("elapsed_seconds=%s seconds_per_step=%s iterations=%s best_step=%s load_count=%s boundary_scale=%s" % (d.get("elapsed_seconds"), d.get("seconds_per_step"), d.get("iterations"), d.get("best_step"), d.get("load_count"), d.get("boundary_scale")))' "$metrics"
  fi
done
