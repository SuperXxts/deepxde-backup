#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
CASE=${CASE:-learnable_A_anchor_reaction}
SCREEN_NAME="bupinn_v2_preflight_${CASE}_${RUN_STAMP}"

cd "$PROJECT"

if [ ! -f hpc/run_v2_preflight_lf.sh ]; then
  echo "ERROR: missing hpc/run_v2_preflight_lf.sh" >&2
  exit 2
fi

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN_NAME" >&2
  exit 2
fi

chmod +x hpc/run_v2_preflight_lf.sh
screen -dmS "$SCREEN_NAME" bash -lc \
  "RUN_STAMP=$RUN_STAMP CASE=$CASE ITERATIONS=1000 DISPLAY_EVERY=100 NUM_DOMAIN=4000 NUM_BOUNDARY=800 OBS_GRID=15 VAL_GRID=41 TEST_GRID=101 ANCHOR_COUNT=8 REACTION_POINTS=200 WIDTH=64 DEPTH=4 SEED=42 DEVICE_INDEX=0 bash hpc/run_v2_preflight_lf.sh"

echo "screen=$SCREEN_NAME"
echo "run_stamp=$RUN_STAMP"
echo "case=$CASE"
echo "run_dir=$PROJECT/exp/00.Smoke/v2_preflight_${CASE}_single_domain4000_boundary800_obs15_${RUN_STAMP}"
