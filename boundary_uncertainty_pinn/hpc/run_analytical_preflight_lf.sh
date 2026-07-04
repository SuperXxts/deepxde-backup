#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
RUN_STAMP=${RUN_STAMP:-manual}
DEVICE_INDEX=${DEVICE_INDEX:-0}
CASE=${CASE:-oracle_A}
ITERATIONS=${ITERATIONS:-30}
DISPLAY_EVERY=${DISPLAY_EVERY:-1}
NUM_DOMAIN=${NUM_DOMAIN:-4000}
NUM_BOUNDARY=${NUM_BOUNDARY:-800}
OBS_GRID=${OBS_GRID:-15}
TEST_GRID=${TEST_GRID:-51}
ANCHOR_COUNT=${ANCHOR_COUNT:-8}
REACTION_POINTS=${REACTION_POINTS:-200}
WIDTH=${WIDTH:-64}
DEPTH=${DEPTH:-4}
SEED=${SEED:-42}

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

RUN_NAME="preflight_${CASE}_domain${NUM_DOMAIN}_boundary${NUM_BOUNDARY}_obs${OBS_GRID}_${RUN_STAMP}"
RUN_DIR="$PROJECT/exp/00.Smoke/$RUN_NAME"
if [ -e "$RUN_DIR" ]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/txt"
{
  echo "===== PREFLIGHT START ====="
  echo "run_dir=$RUN_DIR"
  echo "case=$CASE"
  echo "node=$(hostname)"
  echo "device_index=$DEVICE_INDEX"
  echo "iterations=$ITERATIONS"
  echo "display_every=$DISPLAY_EVERY"
  echo "num_domain=$NUM_DOMAIN"
  echo "num_boundary=$NUM_BOUNDARY"
  echo "obs_grid=$OBS_GRID"
  echo "test_grid=$TEST_GRID"
  echo "anchor_count=$ANCHOR_COUNT"
  echo "reaction_points=$REACTION_POINTS"
  echo "width=$WIDTH"
  echo "depth=$DEPTH"
  echo "seed=$SEED"
  date "+start_time=%F %T"
} | tee "$RUN_DIR/txt/driver.log"

python scripts/train_deepxde_pfnn.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --case "$CASE" \
  --iterations "$ITERATIONS" \
  --display-every "$DISPLAY_EVERY" \
  --num-domain "$NUM_DOMAIN" \
  --num-boundary "$NUM_BOUNDARY" \
  --obs-grid "$OBS_GRID" \
  --test-grid "$TEST_GRID" \
  --anchor-count "$ANCHOR_COUNT" \
  --reaction-points "$REACTION_POINTS" \
  --width "$WIDTH" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/txt/train.log"

{
  echo "===== PREFLIGHT DONE ====="
  date "+end_time=%F %T"
} | tee -a "$RUN_DIR/txt/driver.log"
