#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP=${GROUP:-02.FEMMultiLoadGateSmoke}
NODE=${NODE:-comput1}
DEVICE_INDEX=${DEVICE_INDEX:-0}
RUN_DIR="$PROJECT/exp/$GROUP/C1"
SCREEN_NAME="bupinn_C1_${RUN_STAMP}"

echo "launch case=C1 node=$NODE device=$DEVICE_INDEX run_dir=$RUN_DIR"
ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
  if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
  chmod +x hpc/run_fem_multiload_gate_onecase_lf.sh; \
  screen -dmS '$SCREEN_NAME' env CASE=C1 RUN_NAME=C1 RUN_STAMP='$RUN_STAMP' DEVICE_INDEX='$DEVICE_INDEX' ITERATIONS=3000 DISPLAY_EVERY=1000 CHECKPOINT_EVERY=3000 SEED=42 GROUP='$GROUP' CONDA_ENV=pytorch2.1w2 RUN_NOTE=smoke_three_load_true_boundary_rerun bash hpc/run_fem_multiload_gate_onecase_lf.sh"

echo "run_stamp=$RUN_STAMP"
