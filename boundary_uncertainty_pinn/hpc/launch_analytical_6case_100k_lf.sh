#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="bupinn_analytical6_100k_${RUN_STAMP}"

cd "$PROJECT"

if [ ! -f hpc/run_analytical_6case_100k_lf.sh ]; then
  echo "ERROR: missing hpc/run_analytical_6case_100k_lf.sh" >&2
  exit 2
fi

for CASE in oracle_A wrong_fixed_A learnable_A learnable_A_anchor learnable_A_reaction full_boundary_aware; do
  RUN_DIR="$PROJECT/exp/01.AnalyticalBenchmark/mms_smooth_lens_obs225_A085_seed42_${CASE}_${RUN_STAMP}"
  if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
  fi
done

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN_NAME" >&2
  exit 2
fi

chmod +x hpc/run_analytical_6case_100k_lf.sh
screen -dmS "$SCREEN_NAME" bash -lc \
  "RUN_STAMP=$RUN_STAMP ITERATIONS=100000 DISPLAY_EVERY=500 NUM_DOMAIN=4000 NUM_BOUNDARY=800 OBS_GRID=15 TEST_GRID=201 ANCHOR_COUNT=8 REACTION_POINTS=200 WIDTH=64 DEPTH=4 SEED=42 DEVICE_INDEX=0 bash hpc/run_analytical_6case_100k_lf.sh"

echo "screen=$SCREEN_NAME"
echo "run_stamp=$RUN_STAMP"
echo "first_run_dir=$PROJECT/exp/01.AnalyticalBenchmark/mms_smooth_lens_obs225_A085_seed42_oracle_A_${RUN_STAMP}"
