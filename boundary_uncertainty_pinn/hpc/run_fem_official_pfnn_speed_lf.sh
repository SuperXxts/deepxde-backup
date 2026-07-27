#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

RUN_STAMP=${RUN_STAMP:?Set RUN_STAMP}
RUN_NAME=${RUN_NAME:?Set RUN_NAME}
DEVICE_INDEX=${DEVICE_INDEX:-0}
LOAD_COUNT=${LOAD_COUNT:-1}
BOUNDARY_SCALE=${BOUNDARY_SCALE:-1.0}
ITERATIONS=${ITERATIONS:-3000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-3000}
NUM_DOMAIN=${NUM_DOMAIN:-2000}
NUM_BOUNDARY=${NUM_BOUNDARY:-600}
OBS_COUNT=${OBS_COUNT:-120}
VAL_COUNT=${VAL_COUNT:-500}
TEST_NX=${TEST_NX:-121}
TEST_NY=${TEST_NY:-61}
WIDTH=${WIDTH:-56}
DEPTH=${DEPTH:-4}
SEED=${SEED:-42}
GROUP=${GROUP:-02.FEMOfficialPFNNSpeedSmoke}
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

RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
if [ -e "$RUN_DIR" ]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/txt"
{
  echo "===== START official PFNN speed diagnostic ====="
  echo "run_dir=$RUN_DIR"
  echo "run_stamp=$RUN_STAMP"
  echo "run_name=$RUN_NAME"
  echo "node=$(hostname)"
  echo "device_index=$DEVICE_INDEX"
  echo "conda_env=${CONDA_DEFAULT_ENV:-}"
  echo "python=$(command -v python)"
  echo "deepxde_root=$DEEPXDE_ROOT"
  echo "load_count=$LOAD_COUNT"
  echo "boundary_scale=$BOUNDARY_SCALE"
  echo "iterations=$ITERATIONS"
  echo "display_every=$DISPLAY_EVERY"
  echo "checkpoint_every=$CHECKPOINT_EVERY"
  echo "num_domain=$NUM_DOMAIN"
  echo "num_boundary=$NUM_BOUNDARY"
  echo "obs_count=$OBS_COUNT"
  echo "val_count=$VAL_COUNT"
  echo "test_nx=$TEST_NX"
  echo "test_ny=$TEST_NY"
  echo "width=$WIDTH"
  echo "depth=$DEPTH"
  echo "seed=$SEED"
  echo "fem_data_dir=$FEM_DATA_DIR"
  echo "run_note=$RUN_NOTE"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  date "+start_time=%F %T"
} | tee "$RUN_DIR/txt/driver.log"

python scripts/train_fem_official_pfnn_speed.py \
  --run-dir "$RUN_DIR" \
  --deepxde-root "$DEEPXDE_ROOT" \
  --load-count "$LOAD_COUNT" \
  --boundary-scale "$BOUNDARY_SCALE" \
  --iterations "$ITERATIONS" \
  --display-every "$DISPLAY_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --num-domain "$NUM_DOMAIN" \
  --num-boundary "$NUM_BOUNDARY" \
  --obs-count "$OBS_COUNT" \
  --val-count "$VAL_COUNT" \
  --test-nx "$TEST_NX" \
  --test-ny "$TEST_NY" \
  --width "$WIDTH" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  --run-note "$RUN_NOTE" \
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
  echo "===== DONE official PFNN speed diagnostic ====="
  date "+end_time=%F %T"
} | tee -a "$RUN_DIR/txt/driver.log"
