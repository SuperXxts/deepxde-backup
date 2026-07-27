#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP=${GROUP:-02.FEMThreeLoadFormal_B4B6Smoke}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-3000}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

CASES=(C5 C6 C7)
RUN_NAMES=(B4 B5 B6)
NODES=(comput3 comput3 comput4)
DEVICES=(0 1 0)
RUN_NOTES=(
  B4_three_load_learnable_boundary_reaction_anchor
  B5_three_load_boundary_branch_reaction_anchor
  B6_three_load_full_decoupled_method
)

for i in "${!CASES[@]}"; do
  CASE=${CASES[$i]}
  RUN_NAME=${RUN_NAMES[$i]}
  NODE=${NODES[$i]}
  DEVICE_INDEX=${DEVICES[$i]}
  RUN_NOTE=${RUN_NOTES[$i]}
  RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
  SCREEN_NAME="bupinn_${RUN_NAME}_${RUN_STAMP}"
  remote_env="CASE=$CASE RUN_NAME=$RUN_NAME RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY CHECKPOINT_EVERY=$CHECKPOINT_EVERY SEED=$SEED GROUP=$GROUP CONDA_ENV=$CONDA_ENV RUN_NOTE=$RUN_NOTE"
  echo "launch case=$CASE node=$NODE device=$DEVICE_INDEX run_dir=$RUN_DIR"
  ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
    if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
    if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
    chmod +x hpc/run_fem_multiload_gate_onecase_lf.sh; \
    screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_fem_multiload_gate_onecase_lf.sh'"
done

echo "run_stamp=$RUN_STAMP"
echo "group=$GROUP"
echo "runs=B4 B5 B6"
echo "case_mapping=B4:C5 B5:C6 B6:C7"
echo "eta=smoke: about 20-60 minutes; formal: update after the first 1000-step interval"
