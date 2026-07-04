#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="bupinn_v2_analytical6_150k_${RUN_STAMP}"

cd "$PROJECT"

if [ ! -f hpc/run_v2_analytical_6case_150k_lf.sh ]; then
  echo "ERROR: missing hpc/run_v2_analytical_6case_150k_lf.sh" >&2
  exit 2
fi

for CASE in full_boundary_oracle correct_top_amp_only wrong_fixed_A learnable_A learnable_A_reaction learnable_A_anchor_reaction; do
  RUN_DIR="$PROJECT/exp/01.AnalyticalBenchmark/v2_mms_single_obs225_seed42_${CASE}_${RUN_STAMP}"
  if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
  fi
done

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN_NAME" >&2
  exit 2
fi

chmod +x hpc/run_v2_analytical_6case_150k_lf.sh
screen -dmS "$SCREEN_NAME" bash -lc \
  "RUN_STAMP=$RUN_STAMP ITERATIONS=150000 DISPLAY_EVERY=500 NUM_DOMAIN=4000 NUM_BOUNDARY=800 OBS_GRID=15 VAL_GRID=41 TEST_GRID=201 ANCHOR_COUNT=8 REACTION_POINTS=200 WIDTH=64 DEPTH=4 SEED=42 DEVICE_INDEX=0 bash hpc/run_v2_analytical_6case_150k_lf.sh"

echo "screen=$SCREEN_NAME"
echo "run_stamp=$RUN_STAMP"
echo "first_run_dir=$PROJECT/exp/01.AnalyticalBenchmark/v2_mms_single_obs225_seed42_full_boundary_oracle_${RUN_STAMP}"
