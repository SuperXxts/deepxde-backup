#!/usr/bin/env bash
set -euo pipefail

GROUP=${1:-02.FEMMultiLoadGateSmoke}
ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
cd "$ROOT"

echo "group=$GROUP"
echo "time=$(date '+%F %T')"
echo "== live processes =="
for node in comput1 comput2 comput3 comput4 comput5 comput6 comput7 comput8; do
  echo "-- $node --"
  ssh "$node" "ps -eo pid,user,stat,etime,cmd | grep train_fem_multiload_gate.py | grep -v grep || true"
done

echo "== run status =="
python - "$GROUP" <<'PY'
import json
from pathlib import Path

root = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
import sys
group = sys.argv[1] if len(sys.argv) > 1 else "02.FEMMultiLoadGateSmoke"
for case in [f"C{i}" for i in range(6)]:
    d = root / "exp" / group / case
    log = d / "txt" / "train.log"
    metrics = d / "metrics" / "metrics.json"
    status = d / "json" / "runtime_status.json"
    if not d.exists():
        print(case, "NO_DIR")
        continue
    last_step = "NA"
    err = ""
    if log.exists():
        lines = log.read_text(errors="ignore").splitlines()
        steps = [line for line in lines if line.startswith("STEP ")]
        if steps:
            last_step = steps[-1]
        hits = [line for line in lines if any(k in line.lower() for k in ["traceback", "runtimeerror", "valueerror", "killed", "nan"])]
        if hits:
            err = "ERR=" + hits[-1][:180]
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rel = m.get("relative_l2", {})
        print(case, "DONE", f"K={rel.get('K')}", f"mu={rel.get('mu')}", f"E={rel.get('E')}", f"nu={rel.get('nu')}", f"best={m.get('best_step')}")
    else:
        complete = "status=" + json.loads(status.read_text(encoding="utf-8")).get("status", "") if status.exists() else "RUNNING_OR_FAILED"
        print(case, complete, last_step, err)
PY
