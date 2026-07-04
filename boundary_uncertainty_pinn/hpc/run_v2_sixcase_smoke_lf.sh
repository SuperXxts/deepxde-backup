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
  full_boundary_oracle
  correct_top_amp_only
  wrong_fixed_A
  learnable_A
  learnable_A_reaction
  learnable_A_anchor_reaction
)

for CASE in "${CASES[@]}"; do
  RUN_NAME="v2_single_smoke_${CASE}_${RUN_STAMP}"
  RUN_DIR="$PROJECT/exp/00.Smoke/$RUN_NAME"
  if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR/txt"
  {
    echo "===== START $CASE ====="
    echo "run_dir=$RUN_DIR"
    echo "node=$(hostname)"
    echo "device_index=$DEVICE_INDEX"
    date "+start_time=%F %T"
  } | tee "$RUN_DIR/txt/driver.log"
  python scripts/train_deepxde_pfnn_v2.py \
    --run-dir "$RUN_DIR" \
    --deepxde-root "$DEEPXDE_ROOT" \
    --case "$CASE" \
    --material-mode single \
    --iterations 500 \
    --display-every 100 \
    --num-domain 300 \
    --num-boundary 120 \
    --obs-grid 6 \
    --val-grid 21 \
    --test-grid 51 \
    --anchor-count 4 \
    --reaction-points 80 \
    --width 48 \
    --depth 3 \
    --seed 42 \
    2>&1 | tee "$RUN_DIR/txt/train.log"
  {
    echo "===== DONE $CASE ====="
    date "+end_time=%F %T"
  } | tee -a "$RUN_DIR/txt/driver.log"
done
