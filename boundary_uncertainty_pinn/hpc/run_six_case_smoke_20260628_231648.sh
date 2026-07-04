#!/usr/bin/env bash
set -euo pipefail
PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
CASES=(oracle_A wrong_fixed_A learnable_A learnable_A_anchor learnable_A_reaction full_boundary_aware)
for CASE in "\"; do
  RUN_NAME=sixcase_smoke_\_20260628_231648
  RUN_DIR=\/exp/00.Smoke/\
  echo "===== START \ \ ====="
  DEVICE_INDEX=0 CASE=\ RUN_NAME=\ RUN_DIR=\ ITERATIONS=300 DISPLAY_EVERY=100 NUM_DOMAIN=300 NUM_BOUNDARY=120 OBS_GRID=6 TEST_GRID=51 ANCHOR_COUNT=4 REACTION_POINTS=80 WIDTH=48 DEPTH=3 SEED=42 bash hpc/run_deepxde_smoke.sh
  echo "===== DONE \ \ ====="
done
