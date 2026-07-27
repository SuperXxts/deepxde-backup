#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

CASE=${CASE:?Set CASE, e.g. A15}
RUN_STAMP=${RUN_STAMP:?Set RUN_STAMP, e.g. 20260702_smoke}
DEVICE_INDEX=${DEVICE_INDEX:-0}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5000}
NUM_DOMAIN=${NUM_DOMAIN:-500}
NUM_BOUNDARY=${NUM_BOUNDARY:-160}
OBS_GRID=${OBS_GRID:-8}
VAL_GRID=${VAL_GRID:-21}
TEST_GRID=${TEST_GRID:-51}
OBS_COUNT=${OBS_COUNT:-}
VAL_COUNT=${VAL_COUNT:-}
TEST_NX=${TEST_NX:-}
TEST_NY=${TEST_NY:-}
ANCHOR_COUNT=${ANCHOR_COUNT:-4}
REACTION_POINTS=${REACTION_POINTS:-80}
WIDTH=${WIDTH:-48}
DEPTH=${DEPTH:-3}
MATERIAL_WIDTH=${MATERIAL_WIDTH:-48}
MATERIAL_DEPTH=${MATERIAL_DEPTH:-3}
WIDE_WIDTH=${WIDE_WIDTH:-80}
WIDE_DEPTH=${WIDE_DEPTH:-4}
BOUNDARY_MODES=${BOUNDARY_MODES:-1}
SEED=${SEED:-42}
GROUP=${GROUP:-00.Smoke}
RUN_NAME=${RUN_NAME:-$CASE}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-1.0}
MOMENTUM_WEIGHT=${MOMENTUM_WEIGHT:-}
CONSTITUTIVE_WEIGHT=${CONSTITUTIVE_WEIGHT:-}
BOUNDARY_WEIGHT=${BOUNDARY_WEIGHT:-1.0}
OBS_WEIGHT=${OBS_WEIGHT:-1.0}
OBS_MATERIAL_WEIGHT=${OBS_MATERIAL_WEIGHT:-1.0}
OBS_BOUNDARY_WEIGHT=${OBS_BOUNDARY_WEIGHT:-1.0}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-1.0}
REACTION_WEIGHT=${REACTION_WEIGHT:-1.0}
GRADIENT_DIAGNOSTIC_PERIOD=${GRADIENT_DIAGNOSTIC_PERIOD:-0}
EQUILIBRIUM_FORM=${EQUILIBRIUM_FORM:-mixed}
REACTION_FORM=${REACTION_FORM:-stress}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-best}
RUN_NOTE=${RUN_NOTE:-}
FORMAL_CORE_MAP=${FORMAL_CORE_MAP:-0}

export PATH=/opt/dtk-24.04.3/bin:/opt/dtk-24.04/bin:/usr/local/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:/opt/dtk-24.04/lib64:/opt/dtk-24.04/lib:/opt/dtk-24.04/hip/lib:/opt/dtk-24.04/llvm/lib:${LD_LIBRARY_PATH:-}
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$DEEPXDE_ROOT:$PROJECT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$DEVICE_INDEX
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export DDE_BACKEND=pytorch

cd "$PROJECT"

case "$CASE" in
  A0|A1|A2|A3|A4|A5|A6|A7|A8|A9|A10|A11|A12|A13|A14|A15)
    ;;
  *)
    echo "ERROR: unsupported CASE=$CASE" >&2
    exit 2
    ;;
esac

RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"

if [ -e "$RUN_DIR" ]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/txt"
{
  echo "===== START $CASE ====="
  echo "run_dir=$RUN_DIR"
  echo "run_stamp=$RUN_STAMP"
  echo "case=$CASE"
  echo "run_name=$RUN_NAME"
  echo "node=$(hostname)"
  echo "device_index=$DEVICE_INDEX"
  echo "conda_env=${CONDA_DEFAULT_ENV:-}"
  echo "iterations=$ITERATIONS"
  echo "display_every=$DISPLAY_EVERY"
  echo "checkpoint_every=$CHECKPOINT_EVERY"
  echo "num_domain=$NUM_DOMAIN"
  echo "num_boundary=$NUM_BOUNDARY"
  echo "obs_grid=$OBS_GRID"
  echo "val_grid=$VAL_GRID"
  echo "test_grid=$TEST_GRID"
  echo "obs_count=${OBS_COUNT:-<grid>}"
  echo "val_count=${VAL_COUNT:-<grid>}"
  echo "test_nx=${TEST_NX:-<grid>}"
  echo "test_ny=${TEST_NY:-<grid>}"
  echo "anchor_count=$ANCHOR_COUNT"
  echo "reaction_points=$REACTION_POINTS"
  echo "width=$WIDTH"
  echo "depth=$DEPTH"
  echo "material_width=$MATERIAL_WIDTH"
  echo "material_depth=$MATERIAL_DEPTH"
  echo "wide_width=$WIDE_WIDTH"
  echo "wide_depth=$WIDE_DEPTH"
  echo "boundary_modes=$BOUNDARY_MODES"
  echo "seed=$SEED"
  echo "physics_weight=$PHYSICS_WEIGHT"
  echo "momentum_weight=${MOMENTUM_WEIGHT:-<default>}"
  echo "constitutive_weight=${CONSTITUTIVE_WEIGHT:-<default>}"
  echo "boundary_weight=$BOUNDARY_WEIGHT"
  echo "obs_weight=$OBS_WEIGHT"
  echo "obs_material_weight=$OBS_MATERIAL_WEIGHT"
  echo "obs_boundary_weight=$OBS_BOUNDARY_WEIGHT"
  echo "anchor_weight=$ANCHOR_WEIGHT"
  echo "reaction_weight=$REACTION_WEIGHT"
  echo "gradient_diagnostic_period=$GRADIENT_DIAGNOSTIC_PERIOD"
  echo "equilibrium_form=$EQUILIBRIUM_FORM"
  echo "reaction_form=$REACTION_FORM"
  echo "eval_checkpoint=$EVAL_CHECKPOINT"
  echo "formal_core_map=$FORMAL_CORE_MAP"
  echo "run_note=$RUN_NOTE"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  date "+start_time=%F %T"
} | tee "$RUN_DIR/txt/driver.log"

EXTRA_ARGS=()
if [ -n "$MOMENTUM_WEIGHT" ]; then
  EXTRA_ARGS+=(--momentum-weight "$MOMENTUM_WEIGHT")
fi
if [ -n "$CONSTITUTIVE_WEIGHT" ]; then
  EXTRA_ARGS+=(--constitutive-weight "$CONSTITUTIVE_WEIGHT")
fi
if [ -n "$OBS_COUNT" ]; then
  EXTRA_ARGS+=(--obs-count "$OBS_COUNT")
fi
if [ -n "$VAL_COUNT" ]; then
  EXTRA_ARGS+=(--val-count "$VAL_COUNT")
fi
if [ -n "$TEST_NX" ]; then
  EXTRA_ARGS+=(--test-nx "$TEST_NX")
fi
if [ -n "$TEST_NY" ]; then
  EXTRA_ARGS+=(--test-ny "$TEST_NY")
fi
if [ "$FORMAL_CORE_MAP" = "1" ]; then
  EXTRA_ARGS+=(--formal-core-map)
fi

python scripts/train_decoupled_pfnn.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --case "$CASE" \
  --iterations "$ITERATIONS" \
  --display-every "$DISPLAY_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --num-domain "$NUM_DOMAIN" \
  --num-boundary "$NUM_BOUNDARY" \
  --obs-grid "$OBS_GRID" \
  --val-grid "$VAL_GRID" \
  --test-grid "$TEST_GRID" \
  --anchor-count "$ANCHOR_COUNT" \
  --reaction-points "$REACTION_POINTS" \
  --width "$WIDTH" \
  --depth "$DEPTH" \
  --material-width "$MATERIAL_WIDTH" \
  --material-depth "$MATERIAL_DEPTH" \
  --wide-width "$WIDE_WIDTH" \
  --wide-depth "$WIDE_DEPTH" \
  --boundary-modes "$BOUNDARY_MODES" \
  --physics-weight "$PHYSICS_WEIGHT" \
  "${EXTRA_ARGS[@]}" \
  --boundary-weight "$BOUNDARY_WEIGHT" \
  --obs-weight "$OBS_WEIGHT" \
  --obs-material-weight "$OBS_MATERIAL_WEIGHT" \
  --obs-boundary-weight "$OBS_BOUNDARY_WEIGHT" \
  --anchor-weight "$ANCHOR_WEIGHT" \
  --reaction-weight "$REACTION_WEIGHT" \
  --gradient-diagnostic-period "$GRADIENT_DIAGNOSTIC_PERIOD" \
  --equilibrium-form "$EQUILIBRIUM_FORM" \
  --reaction-form "$REACTION_FORM" \
  --eval-checkpoint "$EVAL_CHECKPOINT" \
  --run-note "$RUN_NOTE" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/txt/train.log"

{
  echo "===== DONE $CASE ====="
  date "+end_time=%F %T"
} | tee -a "$RUN_DIR/txt/driver.log"
