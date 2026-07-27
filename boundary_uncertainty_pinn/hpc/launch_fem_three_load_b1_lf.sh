#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP=${GROUP:-02.FEMThreeLoadFormal}
ITERATIONS=${ITERATIONS:-150000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5000}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

CASE=C2
RUN_NAME=B1
NODE=${NODE:-comput5}
DEVICE_INDEX=${DEVICE_INDEX:-0}
RUN_NOTE=B1_three_load_wrong_fixed_boundary
RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
SCREEN_NAME="bupinn_${RUN_NAME}_${RUN_STAMP}"
remote_env="CASE=$CASE RUN_NAME=$RUN_NAME RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY CHECKPOINT_EVERY=$CHECKPOINT_EVERY SEED=$SEED GROUP=$GROUP CONDA_ENV=$CONDA_ENV RUN_NOTE=$RUN_NOTE"

echo "launch case=$CASE node=$NODE device=$DEVICE_INDEX run_dir=$RUN_DIR"
ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
  if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
  if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
  chmod +x hpc/run_fem_multiload_gate_onecase_lf.sh; \
  screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_fem_multiload_gate_onecase_lf.sh'"

echo "run_stamp=$RUN_STAMP"
echo "group=$GROUP"
echo "runs=B1"
echo "case_mapping=B1:C2"
echo "eta=about 40-70 hours; update after the first live 1000-step interval"
