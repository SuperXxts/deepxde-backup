#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master

CASE=${CASE:?Set CASE, e.g. learnable_A_reaction}
RUN_STAMP=${RUN_STAMP:?Set RUN_STAMP, e.g. 20260629_212009}
DEVICE_INDEX=${DEVICE_INDEX:-0}
ITERATIONS=${ITERATIONS:-150000}
DISPLAY_EVERY=${DISPLAY_EVERY:-500}
NUM_DOMAIN=${NUM_DOMAIN:-4000}
NUM_BOUNDARY=${NUM_BOUNDARY:-800}
OBS_GRID=${OBS_GRID:-15}
VAL_GRID=${VAL_GRID:-41}
TEST_GRID=${TEST_GRID:-201}
ANCHOR_COUNT=${ANCHOR_COUNT:-8}
REACTION_POINTS=${REACTION_POINTS:-200}
WIDTH=${WIDTH:-64}
DEPTH=${DEPTH:-4}
SEED=${SEED:-42}
GROUP=${GROUP:-01.AnalyticalBenchmark}

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

RUN_NAME="v2_mms_single_obs225_seed${SEED}_${CASE}_${RUN_STAMP}"
RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"

case "$CASE" in
  full_boundary_oracle|full_boundary_oracle_reaction|correct_top_amp_only|correct_top_amp_only_reaction|wrong_fixed_A|wrong_fixed_A_reaction|learnable_A|learnable_A_anchor|learnable_A_reaction|learnable_A_anchor_reaction)
    ;;
  *)
    echo "ERROR: unsupported CASE=$CASE" >&2
    exit 2
    ;;
esac

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
  echo "iterations=$ITERATIONS"
  echo "display_every=$DISPLAY_EVERY"
  echo "num_domain=$NUM_DOMAIN"
  echo "num_boundary=$NUM_BOUNDARY"
  echo "obs_grid=$OBS_GRID"
  echo "val_grid=$VAL_GRID"
  echo "test_grid=$TEST_GRID"
  echo "anchor_count=$ANCHOR_COUNT"
  echo "reaction_points=$REACTION_POINTS"
  echo "width=$WIDTH"
  echo "depth=$DEPTH"
  echo "seed=$SEED"
  date "+start_time=%F %T"
} | tee "$RUN_DIR/txt/driver.log"

python scripts/train_deepxde_pfnn_v2.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --case "$CASE" \
  --material-mode single \
  --iterations "$ITERATIONS" \
  --display-every "$DISPLAY_EVERY" \
  --num-domain "$NUM_DOMAIN" \
  --num-boundary "$NUM_BOUNDARY" \
  --obs-grid "$OBS_GRID" \
  --val-grid "$VAL_GRID" \
  --test-grid "$TEST_GRID" \
  --anchor-count "$ANCHOR_COUNT" \
  --reaction-points "$REACTION_POINTS" \
  --width "$WIDTH" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/txt/train.log"

{
  echo "===== DONE $CASE ====="
  date "+end_time=%F %T"
} | tee -a "$RUN_DIR/txt/driver.log"
