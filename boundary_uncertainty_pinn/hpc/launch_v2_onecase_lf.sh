#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
NODE=${NODE:?Set NODE, e.g. comput2}
CASE=${CASE:?Set CASE, e.g. learnable_A_reaction}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
DEVICE_INDEX=${DEVICE_INDEX:-0}
MATERIAL_MODE=${MATERIAL_MODE:-dual}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-300}
NUM_DOMAIN=${NUM_DOMAIN:-500}
NUM_BOUNDARY=${NUM_BOUNDARY:-160}
OBS_GRID=${OBS_GRID:-8}
VAL_GRID=${VAL_GRID:-21}
TEST_GRID=${TEST_GRID:-51}
ANCHOR_COUNT=${ANCHOR_COUNT:-4}
REACTION_POINTS=${REACTION_POINTS:-80}
WIDTH=${WIDTH:-48}
DEPTH=${DEPTH:-3}
SEED=${SEED:-42}
GROUP=${GROUP:-00.Smoke}
RUN_PREFIX=${RUN_PREFIX:-v2}

RUN_NAME="${RUN_PREFIX}_${MATERIAL_MODE}_obs${OBS_GRID}x${OBS_GRID}_seed${SEED}_${CASE}_${RUN_STAMP}"
RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
SCREEN_NAME="bupinn_${RUN_PREFIX}_${MATERIAL_MODE}_${CASE}_${RUN_STAMP}"

remote_env="CASE=$CASE RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX MATERIAL_MODE=$MATERIAL_MODE ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY NUM_DOMAIN=$NUM_DOMAIN NUM_BOUNDARY=$NUM_BOUNDARY OBS_GRID=$OBS_GRID VAL_GRID=$VAL_GRID TEST_GRID=$TEST_GRID ANCHOR_COUNT=$ANCHOR_COUNT REACTION_POINTS=$REACTION_POINTS WIDTH=$WIDTH DEPTH=$DEPTH SEED=$SEED GROUP=$GROUP RUN_PREFIX=$RUN_PREFIX"

ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
  if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
  if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
  chmod +x hpc/run_v2_onecase_lf.sh; \
  screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_v2_onecase_lf.sh'"

echo "node=$NODE"
echo "screen=$SCREEN_NAME"
echo "case=$CASE"
echo "material_mode=$MATERIAL_MODE"
echo "run_stamp=$RUN_STAMP"
echo "run_dir=$RUN_DIR"
echo "eta=smoke about 10-30 minutes; formal 150k about 4-6 hours based on previous cases"
