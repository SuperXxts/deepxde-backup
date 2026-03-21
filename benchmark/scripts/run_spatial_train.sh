#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash benchmark/scripts/run_spatial_train.sh --method pinn --case single_inclusion ..."
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/compute_node_env.sh" >/tmp/codex_compute_env.log 2>&1 || {
  cat /tmp/codex_compute_env.log
  exit 1
}
cat /tmp/codex_compute_env.log

cd "$REPO_ROOT/../deepxde-master/my_example/spatial_material_inverse"
python train_material_field.py "$@"
