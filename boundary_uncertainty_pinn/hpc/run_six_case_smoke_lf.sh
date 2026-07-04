#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
RUN_STAMP=${RUN_STAMP:-manual}
DEVICE_INDEX=${DEVICE_INDEX:-0}

export PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:${LD_LIBRARY_PATH:-}
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh
conda activate pytorch2.1w2
export CUDA_VISIBLE_DEVICES=$DEVICE_INDEX
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export DDE_BACKEND=pytorch

cd "$PROJECT"

CASES=(
  oracle_A
  wrong_fixed_A
  learnable_A
  learnable_A_anchor
  learnable_A_reaction
  full_boundary_aware
)

for CASE in "${CASES[@]}"; do
  RUN_NAME="sixcase_smoke_${CASE}_${RUN_STAMP}"
  RUN_DIR="$PROJECT/exp/00.Smoke/$RUN_NAME"
  mkdir -p "$RUN_DIR/txt"
  echo "===== START $CASE $RUN_DIR =====" | tee "$RUN_DIR/txt/driver.log"
  python scripts/train_deepxde_pfnn.py \
    --run-dir "$RUN_DIR" \
    --deepxde-root "$DEEPXDE_ROOT" \
    --case "$CASE" \
    --iterations 300 \
    --display-every 100 \
    --num-domain 300 \
    --num-boundary 120 \
    --obs-grid 6 \
    --test-grid 51 \
    --anchor-count 4 \
    --reaction-points 80 \
    --width 48 \
    --depth 3 \
    --seed 42 \
    2>&1 | tee "$RUN_DIR/txt/train.log"
  echo "===== DONE $CASE $RUN_DIR =====" | tee -a "$RUN_DIR/txt/driver.log"
done
