#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="bupinn_v2_smoke_${RUN_STAMP}"

cd "$PROJECT"

if [ ! -f hpc/run_v2_sixcase_smoke_lf.sh ]; then
  echo "ERROR: missing hpc/run_v2_sixcase_smoke_lf.sh" >&2
  exit 2
fi

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN_NAME" >&2
  exit 2
fi

chmod +x hpc/run_v2_sixcase_smoke_lf.sh
screen -dmS "$SCREEN_NAME" bash -lc "RUN_STAMP=$RUN_STAMP DEVICE_INDEX=0 bash hpc/run_v2_sixcase_smoke_lf.sh"

echo "screen=$SCREEN_NAME"
echo "run_stamp=$RUN_STAMP"
echo "first_run_dir=$PROJECT/exp/00.Smoke/v2_single_smoke_full_boundary_oracle_${RUN_STAMP}"
