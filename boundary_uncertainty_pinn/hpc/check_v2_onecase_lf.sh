#!/usr/bin/env bash
set -euo pipefail

NODE=${1:?Usage: check_v2_onecase_lf.sh NODE RUN_DIR [WAIT_SECONDS]}
RUN_DIR=${2:?Usage: check_v2_onecase_lf.sh NODE RUN_DIR [WAIT_SECONDS]}
WAIT_SECONDS=${3:-30}

ssh "$NODE" bash -s -- "$RUN_DIR" "$WAIT_SECONDS" <<'REMOTE'
set -euo pipefail
RUN_DIR=${1:?}
WAIT_SECONDS=${2:?}
LOG="$RUN_DIR/txt/train.log"
CONFIG="$RUN_DIR/json/config.json"
METRICS="$RUN_DIR/metrics/metrics.json"

echo "===== LIVE CHECK ====="
echo "node=$(hostname)"
date "+check_time=%F %T"
echo "run_dir=$RUN_DIR"
echo "wait_seconds=$WAIT_SECONDS"

echo "===== PROCESS ====="
ps -u "${USER:-wxtian}" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= \
  | grep train_deepxde_pfnn_v2.py \
  | grep "$RUN_DIR" \
  | grep -v grep || true

echo "===== LOG BEFORE ====="
if [ ! -f "$LOG" ]; then
  echo "missing_log=$LOG"
  exit 0
fi
stat -c "mtime_before=%y size_before=%s" "$LOG"
STEP_BEFORE=$(grep -E "STEP [0-9]+" "$LOG" | tail -1 || true)
echo "step_before=$STEP_BEFORE"
grep -E "RUN_COMPLETE|Traceback|Error|Exception|Compiling|Training model|STEP [0-9]+" "$LOG" | tail -20 || true

sleep "$WAIT_SECONDS"

echo "===== LOG AFTER ====="
stat -c "mtime_after=%y size_after=%s" "$LOG"
STEP_AFTER=$(grep -E "STEP [0-9]+" "$LOG" | tail -1 || true)
echo "step_after=$STEP_AFTER"
grep -E "RUN_COMPLETE|Traceback|Error|Exception|STEP [0-9]+" "$LOG" | tail -20 || true

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

echo "===== CONFIG ====="
if [ -f "$CONFIG" ]; then
  python - "$CONFIG" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = json.load(f)
for key in [
    "case",
    "material_mode",
    "material_parameterization",
    "backend",
    "torch_cuda_available",
    "torch_device_count",
]:
    print(f"{key}={c.get(key)}")
PY
else
  echo "missing_config=$CONFIG"
fi

echo "===== METRICS ====="
if [ -f "$METRICS" ]; then
  python - "$METRICS" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    m = json.load(f)
print(f"completed_case={m.get('case')}")
print(f"final_A={m.get('final_A')}")
print(f"reaction_rel_error={m.get('reaction_rel_error')}")
rel = m.get("relative_l2", {})
for key in ["K", "mu", "E", "nu", "ux", "uy"]:
    if key in rel:
        print(f"rel_{key}={rel[key]}")
PY
else
  echo "missing_metrics=$METRICS"
fi
REMOTE
