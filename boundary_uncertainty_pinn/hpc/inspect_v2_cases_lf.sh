#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
GROUP=${GROUP:-01.AnalyticalBenchmark}
RUN_STAMP=${RUN_STAMP:?Set RUN_STAMP}
SEED=${SEED:-42}

if [ "$#" -lt 1 ]; then
  echo "Usage: GROUP=... RUN_STAMP=... $0 case [case ...]" >&2
  exit 2
fi

cd "$PROJECT"

for case_name in "$@"; do
  run_name="v2_mms_single_obs225_seed${SEED}_${case_name}_${RUN_STAMP}"
  run_dir="$PROJECT/exp/$GROUP/$run_name"
  log_file="$run_dir/txt/train.log"

  echo "===== $case_name ====="
  echo "run_dir=$run_dir"
  if [ ! -d "$run_dir" ]; then
    echo "dir=missing"
    continue
  fi
  echo "dir=exists"

  echo "-- live_process --"
  pgrep -af "train_deepxde_pfnn_v2.py.*${case_name}.*${RUN_STAMP}" || true

  echo "-- log_status --"
  if [ -f "$log_file" ]; then
    stat -c "train_log_mtime=%y" "$log_file"
    grep "STEP " "$log_file" | tail -3 || true
    grep "RUN_COMPLETE" "$log_file" | tail -1 || true
    tail -5 "$log_file" || true
  else
    echo "train_log=missing"
  fi

  echo "-- metrics --"
  if [ -f "$run_dir/metrics/metrics.json" ]; then
    python - "$run_dir/metrics/metrics.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
rel = metrics.get("relative_l2", {})
print(f"case={metrics.get('case')}")
print(f"iterations={metrics.get('iterations')}")
print(f"final_A={metrics.get('final_A')}")
print(f"reaction_rel_error={metrics.get('reaction_rel_error')}")
print(f"elapsed_seconds={metrics.get('elapsed_seconds')}")
print(
    "relative_l2="
    + ", ".join(
        f"{name}:{rel.get(name)}"
        for name in ("ux", "uy", "K", "mu", "E", "m")
        if name in rel
    )
)
PY
  else
    echo "metrics=missing"
  fi

  echo "-- artifacts --"
  for subdir in txt png metrics json model npz dat; do
    if [ -d "$run_dir/$subdir" ]; then
      count=$(find "$run_dir/$subdir" -type f | wc -l)
      echo "$subdir=$count"
    else
      echo "$subdir=missing"
    fi
  done
done
