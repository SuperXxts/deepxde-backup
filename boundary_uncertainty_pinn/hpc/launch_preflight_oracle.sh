#!/usr/bin/env bash
set -euo pipefail
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
SCREEN_NAME=bupinn_preflight_oracle_4000_20260628_235909
RUN_STAMP=20260628_235909
if screen -ls | grep -q "[.]bupinn_preflight_oracle_4000_20260628_235909"; then
  echo "ERROR: screen already exists: bupinn_preflight_oracle_4000_20260628_235909" >&2
  exit 2
fi
cd ""
chmod +x hpc/run_analytical_preflight_lf.sh
screen -dmS "bupinn_preflight_oracle_4000_20260628_235909" bash -lc "RUN_STAMP=20260628_235909 CASE=oracle_A ITERATIONS=30 DISPLAY_EVERY=1 NUM_DOMAIN=4000 NUM_BOUNDARY=800 OBS_GRID=15 TEST_GRID=51 ANCHOR_COUNT=8 REACTION_POINTS=200 WIDTH=64 DEPTH=4 SEED=42 DEVICE_INDEX=0 bash hpc/run_analytical_preflight_lf.sh"
echo "screen=bupinn_preflight_oracle_4000_20260628_235909"
echo "run_stamp=20260628_235909"
echo "run_dir=/exp/00.Smoke/preflight_oracle_A_domain4000_boundary800_obs15_20260628_235909"