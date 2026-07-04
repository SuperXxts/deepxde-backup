#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="bupinn_preflight_oracle_4000_${RUN_STAMP}"
RUN_DIR="$PROJECT/exp/00.Smoke/preflight_oracle_A_domain4000_boundary800_obs15_${RUN_STAMP}"

cd "$PROJECT"

if [ ! -f hpc/run_analytical_preflight_lf.sh ]; then
  echo "ERROR: missing hpc/run_analytical_preflight_lf.sh" >&2
  exit 2
fi

if [ -e "$RUN_DIR" ]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN_NAME" >&2
  exit 2
fi

chmod +x hpc/run_analytical_preflight_lf.sh
screen -dmS "$SCREEN_NAME" bash -lc \
  "RUN_STAMP=$RUN_STAMP CASE=oracle_A ITERATIONS=30 DISPLAY_EVERY=1 NUM_DOMAIN=4000 NUM_BOUNDARY=800 OBS_GRID=15 TEST_GRID=51 ANCHOR_COUNT=8 REACTION_POINTS=200 WIDTH=64 DEPTH=4 SEED=42 DEVICE_INDEX=0 bash hpc/run_analytical_preflight_lf.sh"

echo "screen=$SCREEN_NAME"
echo "run_stamp=$RUN_STAMP"
echo "run_dir=$RUN_DIR"
