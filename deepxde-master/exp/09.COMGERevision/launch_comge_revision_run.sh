#!/usr/bin/env bash
set -euo pipefail

REPO=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$REPO/my_example/spatial_material_inverse

: "${CASE:?CASE is required}"
: "${METHOD:?METHOD is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${SAVE_DIR:?SAVE_DIR is required}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-100000}"
DISPLAY_EVERY="${DISPLAY_EVERY:-1000}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
LR="${LR:-0.0005}"
NUM_DOMAIN="${NUM_DOMAIN:-1024}"
NUM_TEST="${NUM_TEST:-512}"
NUM_BOUNDARY="${NUM_BOUNDARY:-160}"
NUM_OBSERVE="${NUM_OBSERVE:-1200}"
NUM_VAL_OBSERVE="${NUM_VAL_OBSERVE:-1200}"
NUM_EVAL_OBSERVE="${NUM_EVAL_OBSERVE:-1200}"
DATA_WEIGHT="${DATA_WEIGHT:-20.0}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-20.0}"
REG_WEIGHT="${REG_WEIGHT:-0.0001}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-128,128,128,128}"
STATE_LAYERS="${STATE_LAYERS:-128,128,128,128}"
MATERIAL_LAYERS="${MATERIAL_LAYERS:-128,128,128,128}"
INTERFACE_LAYERS="${INTERFACE_LAYERS:-128,128,128,128}"
GEOMETRY_LAYERS="${GEOMETRY_LAYERS:-64,64,64}"
DYNAMIC_LOSS_BALANCE="${DYNAMIC_LOSS_BALANCE:-none}"
DYNAMIC_SCOPE="${DYNAMIC_SCOPE:-material_branch}"
DYNAMIC_PERIOD="${DYNAMIC_PERIOD:-1000}"
OBSERVATION_SPLIT_TAG="${OBSERVATION_SPLIT_TAG:-revision_uniform_v1}"
OBSERVATION_SPLIT_SEED="${OBSERVATION_SPLIT_SEED:-1059}"
EXP_ROOT="${EXP_ROOT:-$REPO/exp/09.COMGERevision}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-Revision}"
LOAD_SCALES="${LOAD_SCALES:-1.0,1.0,1.0}"
LOAD_MODES="${LOAD_MODES:-normal_to_layer,bending_y,biaxial_bulk}"
EVAL_NX="${EVAL_NX:-81}"
EVAL_NY="${EVAL_NY:-81}"
TEACHER_SOURCE="${TEACHER_SOURCE:-analytic}"

mkdir -p "$SAVE_DIR"/{png,json,metrics,model,txt,npz}
LOG_FILE=$SAVE_DIR/screen.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] $(date '+%F %T %Z')"
echo "[host] $(hostname)"
echo "[case] $CASE"
echo "[method] $METHOD"
echo "[seed] $SEED"
echo "[run] $RUN_NAME"
echo "[save_dir] $SAVE_DIR"

export PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:${LD_LIBRARY_PATH:-}
set +u
source /etc/profile || true
source ~/.bashrc || true
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh || true
set -u
conda activate pytorch2.1w2

export CUDA_VISIBLE_DEVICES="$DEVICE_INDEX"
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO:${PYTHONPATH:-}

python - <<'PY'
import torch
import deepxde
print(f"[runtime] torch={torch.__version__} cuda={torch.cuda.is_available()} count={torch.cuda.device_count()}")
print(f"[runtime] deepxde_backend={deepxde.backend.backend_name}")
PY

cd "$WORK"
python -u train_material_field.py \
  --method "$METHOD" \
  --case "$CASE" \
  --seed "$SEED" \
  --lr "$LR" \
  --iterations "$ITERATIONS" \
  --display_every "$DISPLAY_EVERY" \
  --num_domain "$NUM_DOMAIN" \
  --num_test "$NUM_TEST" \
  --num_boundary "$NUM_BOUNDARY" \
  --num_observe "$NUM_OBSERVE" \
  --num_val_observe "$NUM_VAL_OBSERVE" \
  --num_eval_observe "$NUM_EVAL_OBSERVE" \
  --noise_level 0.0 \
  --reg_weight "$REG_WEIGHT" \
  --data_weight "$DATA_WEIGHT" \
  --boundary_weight "$BOUNDARY_WEIGHT" \
  --lambda_floor 0.1 \
  --mu_floor 0.1 \
  --activation tanh \
  --backbone_type mlp \
  --hidden_layers "$HIDDEN_LAYERS" \
  --state_layers "$STATE_LAYERS" \
  --material_layers "$MATERIAL_LAYERS" \
  --geometry_layers "$GEOMETRY_LAYERS" \
  --interface_layers "$INTERFACE_LAYERS" \
  --interface_sharpness 10.0 \
  --num_frequencies 4 \
  --material_parameterization bulkmu \
  --k_floor 0.2 \
  --load_scales "$LOAD_SCALES" \
  --load_modes "$LOAD_MODES" \
  --load_balance_mode reference_rms \
  --load_balance_nx 48 \
  --load_balance_ny 48 \
  --load_balance_epsilon 1e-6 \
  --primary_load_index 0 \
  --observation_split_tag "$OBSERVATION_SPLIT_TAG" \
  --observation_cache_dir "$REPO/exp/_shared_observation_splits/spatial_material_inverse" \
  --observation_split_seed "$OBSERVATION_SPLIT_SEED" \
  --reaction_weight 0.0 \
  --reaction_edge top \
  --exp_root "$EXP_ROOT" \
  --experiment_group "$EXPERIMENT_GROUP" \
  --run_name "$RUN_NAME" \
  --save_dir "$SAVE_DIR" \
  --eval_nx "$EVAL_NX" \
  --eval_ny "$EVAL_NY" \
  --run_eval_after_train \
  --dynamic_loss_balance "$DYNAMIC_LOSS_BALANCE" \
  --dynamic_balance_period "$DYNAMIC_PERIOD" \
  --dynamic_balance_parameter_scope "$DYNAMIC_SCOPE" \
  --dynamic_balance_ema 0.5 \
  --dynamic_balance_reg_scale 1.0 \
  --dynamic_balance_min_physics_scale 0.25 \
  --dynamic_balance_max_physics_scale 4.0 \
  --dynamic_balance_min_observation_scale 0.25 \
  --dynamic_balance_max_observation_scale 4.0 \
  --dynamic_balance_min_boundary_scale 0.25 \
  --dynamic_balance_max_boundary_scale 4.0 \
  --dynamic_balance_grad_eps 1e-12 \
  --validation_observation_target clean \
  --report_checkpoint best_model \
  --teacher_source "$TEACHER_SOURCE"

echo "[end] $(date '+%F %T %Z') status=$?"
