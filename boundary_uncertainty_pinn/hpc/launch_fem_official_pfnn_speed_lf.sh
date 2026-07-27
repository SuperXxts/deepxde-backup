#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP=${GROUP:-02.FEMOfficialPFNNSpeedSmoke}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-3000}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

RUN_NAMES=(P0 P1 P2)
LOAD_COUNTS=(1 3 3)
BOUNDARY_SCALES=(1.0 1.0 0.85)
NODES=(comput5 comput5 comput6)
DEVICES=(0 1 0)
RUN_NOTES=(
  official_pfnn_single_load_true_boundary_same_fem_data
  official_pfnn_three_load_true_boundary_same_fem_data
  official_pfnn_three_load_wrong_fixed_boundary_same_fem_data
)

for i in "${!RUN_NAMES[@]}"; do
  RUN_NAME=${RUN_NAMES[$i]}
  LOAD_COUNT=${LOAD_COUNTS[$i]}
  BOUNDARY_SCALE=${BOUNDARY_SCALES[$i]}
  NODE=${NODES[$i]}
  DEVICE_INDEX=${DEVICES[$i]}
  RUN_NOTE=${RUN_NOTES[$i]}
  RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
  SCREEN_NAME="bupinn_pfnn_${RUN_NAME}_${RUN_STAMP}"
  remote_env="RUN_NAME=$RUN_NAME RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX LOAD_COUNT=$LOAD_COUNT BOUNDARY_SCALE=$BOUNDARY_SCALE ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY CHECKPOINT_EVERY=$CHECKPOINT_EVERY SEED=$SEED GROUP=$GROUP CONDA_ENV=$CONDA_ENV RUN_NOTE=$RUN_NOTE"
  echo "launch run=$RUN_NAME load_count=$LOAD_COUNT boundary_scale=$BOUNDARY_SCALE node=$NODE device=$DEVICE_INDEX run_dir=$RUN_DIR"
  ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
    if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
    if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
    chmod +x hpc/run_fem_official_pfnn_speed_lf.sh; \
    screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_fem_official_pfnn_speed_lf.sh'"
done

echo "run_stamp=$RUN_STAMP"
echo "group=$GROUP"
echo "runs=P0 P1 P2"
echo "P0=official PFNN, one load, true boundary"
echo "P1=official PFNN, three loads, true boundary"
echo "P2=official PFNN, three loads, wrong fixed boundary scale 0.85"
echo "eta=about 15-35 minutes for 3000-step speed smoke; refine after first 1000-step interval"
