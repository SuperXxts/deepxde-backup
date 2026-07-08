#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

CASE=${CASE:?Set CASE, e.g. C0}
RUN_STAMP=${RUN_STAMP:?Set RUN_STAMP}
DEVICE_INDEX=${DEVICE_INDEX:-0}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
DYNAMIC_FIGURE_EVERY=${DYNAMIC_FIGURE_EVERY:-$DISPLAY_EVERY}
MATERIAL_MONITOR_EVERY=${MATERIAL_MONITOR_EVERY:-$DISPLAY_EVERY}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5000}
NUM_DOMAIN=${NUM_DOMAIN:-2000}
NUM_BOUNDARY=${NUM_BOUNDARY:-600}
OBS_COUNT=${OBS_COUNT:-120}
VAL_COUNT=${VAL_COUNT:-500}
TEST_NX=${TEST_NX:-121}
TEST_NY=${TEST_NY:-61}
ANCHOR_COUNT=${ANCHOR_COUNT:-4}
REACTION_POINTS=${REACTION_POINTS:-80}
WIDTH=${WIDTH:-56}
DEPTH=${DEPTH:-4}
MATERIAL_WIDTH=${MATERIAL_WIDTH:-56}
MATERIAL_DEPTH=${MATERIAL_DEPTH:-4}
BOUNDARY_MODES=${BOUNDARY_MODES:-1}
SEED=${SEED:-42}
GROUP=${GROUP:-02.FEMMultiLoadGateSmoke}
RUN_NAME=${RUN_NAME:-$CASE}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-1.0}
BOUNDARY_WEIGHT=${BOUNDARY_WEIGHT:-1.0}
OBS_WEIGHT=${OBS_WEIGHT:-1.0}
OBS_MATERIAL_WEIGHT=${OBS_MATERIAL_WEIGHT:-20.0}
OBS_BOUNDARY_WEIGHT=${OBS_BOUNDARY_WEIGHT:-20.0}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-1.0}
REACTION_WEIGHT=${REACTION_WEIGHT:-5.0}
MATERIAL_SMOOTHNESS_WEIGHT=${MATERIAL_SMOOTHNESS_WEIGHT:-0.0}
K_MIN_FACTOR=${K_MIN_FACTOR:-0.35}
K_MAX_FACTOR=${K_MAX_FACTOR:-1.80}
MU_MIN_FACTOR=${MU_MIN_FACTOR:-0.40}
MU_MAX_FACTOR=${MU_MAX_FACTOR:-1.70}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-best}
RUN_NOTE=${RUN_NOTE:-}
FEM_NX=${FEM_NX:-96}
FEM_NY=${FEM_NY:-48}
DOMAIN_WIDTH=${DOMAIN_WIDTH:-6.0}
DOMAIN_DEPTH=${DOMAIN_DEPTH:-3.0}
PLATE_WIDTH=${PLATE_WIDTH:-1.0}
SETTLEMENT=${SETTLEMENT:--0.05}
HORIZONTAL_DISPLACEMENT=${HORIZONTAL_DISPLACEMENT:-0.035}
K_REF=${K_REF:-1.0}
MU_REF=${MU_REF:-0.45}
FEM_DATA_DIR=${FEM_DATA_DIR:-$PROJECT/exp/02.FEMMultiLoadGate/_fem_multiload_data}

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

HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
if [ "$HOSTNAME_SHORT" = "admin" ] || [ "$HOSTNAME_SHORT" = "manage" ]; then
  echo "ERROR: this training wrapper must run on a compute node, got host=$HOSTNAME_SHORT" >&2
  exit 3
fi

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
  echo "python=$(command -v python)"
  echo "deepxde_root=$DEEPXDE_ROOT"
  echo "iterations=$ITERATIONS"
  echo "display_every=$DISPLAY_EVERY"
  echo "dynamic_figure_every=$DYNAMIC_FIGURE_EVERY"
  echo "material_monitor_every=$MATERIAL_MONITOR_EVERY"
  echo "checkpoint_every=$CHECKPOINT_EVERY"
  echo "num_domain=$NUM_DOMAIN"
  echo "num_boundary=$NUM_BOUNDARY"
  echo "obs_count=$OBS_COUNT"
  echo "val_count=$VAL_COUNT"
  echo "test_nx=$TEST_NX"
  echo "test_ny=$TEST_NY"
  echo "anchor_count=$ANCHOR_COUNT"
  echo "reaction_points=$REACTION_POINTS"
  echo "width=$WIDTH"
  echo "depth=$DEPTH"
  echo "material_width=$MATERIAL_WIDTH"
  echo "material_depth=$MATERIAL_DEPTH"
  echo "boundary_modes=$BOUNDARY_MODES"
  echo "seed=$SEED"
  echo "physics_weight=$PHYSICS_WEIGHT"
  echo "boundary_weight=$BOUNDARY_WEIGHT"
  echo "obs_weight=$OBS_WEIGHT"
  echo "obs_material_weight=$OBS_MATERIAL_WEIGHT"
  echo "obs_boundary_weight=$OBS_BOUNDARY_WEIGHT"
  echo "anchor_weight=$ANCHOR_WEIGHT"
  echo "reaction_weight=$REACTION_WEIGHT"
  echo "material_smoothness_weight=$MATERIAL_SMOOTHNESS_WEIGHT"
  echo "fem_data_dir=$FEM_DATA_DIR"
  echo "run_note=$RUN_NOTE"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  date "+start_time=%F %T"
} | tee "$RUN_DIR/txt/driver.log"

python scripts/train_fem_multiload_gate.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --case "$CASE" \
  --iterations "$ITERATIONS" \
  --display-every "$DISPLAY_EVERY" \
  --dynamic-figure-every "$DYNAMIC_FIGURE_EVERY" \
  --material-monitor-every "$MATERIAL_MONITOR_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --num-domain "$NUM_DOMAIN" \
  --num-boundary "$NUM_BOUNDARY" \
  --obs-count "$OBS_COUNT" \
  --val-count "$VAL_COUNT" \
  --test-nx "$TEST_NX" \
  --test-ny "$TEST_NY" \
  --anchor-count "$ANCHOR_COUNT" \
  --reaction-points "$REACTION_POINTS" \
  --width "$WIDTH" \
  --depth "$DEPTH" \
  --material-width "$MATERIAL_WIDTH" \
  --material-depth "$MATERIAL_DEPTH" \
  --boundary-modes "$BOUNDARY_MODES" \
  --physics-weight "$PHYSICS_WEIGHT" \
  --boundary-weight "$BOUNDARY_WEIGHT" \
  --obs-weight "$OBS_WEIGHT" \
  --obs-material-weight "$OBS_MATERIAL_WEIGHT" \
  --obs-boundary-weight "$OBS_BOUNDARY_WEIGHT" \
  --anchor-weight "$ANCHOR_WEIGHT" \
  --reaction-weight "$REACTION_WEIGHT" \
  --material-smoothness-weight "$MATERIAL_SMOOTHNESS_WEIGHT" \
  --k-min-factor "$K_MIN_FACTOR" \
  --k-max-factor "$K_MAX_FACTOR" \
  --mu-min-factor "$MU_MIN_FACTOR" \
  --mu-max-factor "$MU_MAX_FACTOR" \
  --eval-checkpoint "$EVAL_CHECKPOINT" \
  --run-note "$RUN_NOTE" \
  --seed "$SEED" \
  --fem-data-dir "$FEM_DATA_DIR" \
  --fem-nx "$FEM_NX" \
  --fem-ny "$FEM_NY" \
  --domain-width "$DOMAIN_WIDTH" \
  --domain-depth "$DOMAIN_DEPTH" \
  --plate-width "$PLATE_WIDTH" \
  --settlement "$SETTLEMENT" \
  --horizontal-displacement "$HORIZONTAL_DISPLACEMENT" \
  --k-ref "$K_REF" \
  --mu-ref "$MU_REF" \
  2>&1 | tee "$RUN_DIR/txt/train.log"

{
  echo "===== DONE $CASE ====="
  date "+end_time=%F %T"
} | tee -a "$RUN_DIR/txt/driver.log"
