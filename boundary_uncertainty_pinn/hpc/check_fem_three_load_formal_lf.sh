#!/usr/bin/env bash
set -euo pipefail

GROUP=${1:-02.FEMThreeLoadFormal}
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
import sys
from pathlib import Path

root = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
group = sys.argv[1] if len(sys.argv) > 1 else "02.FEMThreeLoadFormal"
case_map = {
    "B0": "C1 true boundary, three loads",
    "B1": "C2 wrong fixed boundary, three loads",
    "B2": "C3 learnable boundary, three loads",
    "B3": "C4 learnable boundary + reaction, three loads",
    "B4": "C5 learnable boundary + reaction + anchors, three loads",
    "B5": "C6 boundary branch + reaction + anchors, three loads",
    "B6": "C7 full decoupled method, three loads",
}
for run_name, label in case_map.items():
    d = root / "exp" / group / run_name
    log = d / "txt" / "train.log"
    metrics = d / "metrics" / "metrics.json"
    status = d / "json" / "runtime_status.json"
    if not d.exists():
        print(run_name, "NO_DIR", label)
        continue
    last_step = "NA"
    err = ""
    log_mtime = "NA"
    if log.exists():
        log_mtime = str(int(log.stat().st_mtime))
        lines = log.read_text(errors="ignore").splitlines()
        steps = [line for line in lines if line.startswith("STEP ")]
        if steps:
            last_step = steps[-1]
        hits = [
            line for line in lines
            if any(k in line.lower() for k in ["traceback", "runtimeerror", "valueerror", "killed", "nan"])
        ]
        if hits:
            err = "ERR=" + hits[-1][:180]
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rel = m.get("relative_l2", {})
        print(
            run_name,
            "DONE",
            f"K={rel.get('K')}",
            f"mu={rel.get('mu')}",
            f"E={rel.get('E')}",
            f"nu={rel.get('nu')}",
            f"best={m.get('best_step')}",
            label,
        )
    else:
        state = "RUNNING_OR_FAILED"
        if status.exists():
            state = "status=" + json.loads(status.read_text(encoding="utf-8")).get("status", "")
        print(run_name, state, last_step, f"log_mtime={log_mtime}", err, label)
PY
