#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
DEVICE_INDEX=${DEVICE_INDEX:-0}
CASE=${CASE:-learnable_A_anchor}
RUN_NAME=${RUN_NAME:-smoke_${CASE}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$PROJECT/exp/00.Smoke/$RUN_NAME}

export PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:${LD_LIBRARY_PATH:-}
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh
conda activate pytorch2.1w2
export CUDA_VISIBLE_DEVICES=$DEVICE_INDEX
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export DDE_BACKEND=pytorch

mkdir -p "$RUN_DIR/txt"
cd "$PROJECT"

python scripts/train_deepxde_pfnn.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --case "$CASE" \
  --iterations "${ITERATIONS:-800}" \
  --display-every "${DISPLAY_EVERY:-100}" \
  --num-domain "${NUM_DOMAIN:-500}" \
  --num-boundary "${NUM_BOUNDARY:-160}" \
  --obs-grid "${OBS_GRID:-8}" \
  --test-grid "${TEST_GRID:-101}" \
  --anchor-count "${ANCHOR_COUNT:-4}" \
  --reaction-points "${REACTION_POINTS:-100}" \
  --width "${WIDTH:-64}" \
  --depth "${DEPTH:-4}" \
  --seed "${SEED:-42}" \
  2>&1 | tee "$RUN_DIR/txt/train.log"
