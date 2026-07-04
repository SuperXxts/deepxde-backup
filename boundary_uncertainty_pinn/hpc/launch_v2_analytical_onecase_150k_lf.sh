#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
NODE=${NODE:?Set NODE, e.g. comput2}
CASE=${CASE:?Set CASE, e.g. learnable_A_reaction}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
DEVICE_INDEX=${DEVICE_INDEX:-0}
ITERATIONS=${ITERATIONS:-150000}
DISPLAY_EVERY=${DISPLAY_EVERY:-500}
NUM_DOMAIN=${NUM_DOMAIN:-4000}
NUM_BOUNDARY=${NUM_BOUNDARY:-800}
OBS_GRID=${OBS_GRID:-15}
VAL_GRID=${VAL_GRID:-41}
TEST_GRID=${TEST_GRID:-201}
ANCHOR_COUNT=${ANCHOR_COUNT:-8}
REACTION_POINTS=${REACTION_POINTS:-200}
WIDTH=${WIDTH:-64}
DEPTH=${DEPTH:-4}
SEED=${SEED:-42}
GROUP=${GROUP:-01.AnalyticalBenchmark}

RUN_NAME="v2_mms_single_obs225_seed${SEED}_${CASE}_${RUN_STAMP}"
RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
SCREEN_NAME="bupinn_v2_150k_${CASE}_${RUN_STAMP}"

remote_env="CASE=$CASE RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY NUM_DOMAIN=$NUM_DOMAIN NUM_BOUNDARY=$NUM_BOUNDARY OBS_GRID=$OBS_GRID VAL_GRID=$VAL_GRID TEST_GRID=$TEST_GRID ANCHOR_COUNT=$ANCHOR_COUNT REACTION_POINTS=$REACTION_POINTS WIDTH=$WIDTH DEPTH=$DEPTH SEED=$SEED GROUP=$GROUP"

ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
  if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
  if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
  chmod +x hpc/run_v2_analytical_onecase_150k_lf.sh; \
  screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_v2_analytical_onecase_150k_lf.sh'"

echo "node=$NODE"
echo "screen=$SCREEN_NAME"
echo "case=$CASE"
echo "run_stamp=$RUN_STAMP"
echo "run_dir=$RUN_DIR"
echo "eta=about 4 hours based on previous 150k cases on comput1"
