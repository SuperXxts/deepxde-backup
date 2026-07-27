#!/usr/bin/env bash
set -euo pipefail

REPO=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$REPO/my_example/spatial_material_inverse

: "${CASE:?CASE is required}"
: "${SCOPE:?SCOPE is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${SAVE_DIR:?SAVE_DIR is required}"

mkdir -p "$SAVE_DIR"/{png,json,metrics,model,txt,npz}
LOG_FILE=$SAVE_DIR/screen.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] $(date '+%F %T %Z')"
echo "[host] $(hostname)"
echo "[case] $CASE"
echo "[scope] $SCOPE"
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

export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX:-0}"
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
  --method threebranch_stress_kmu \
  --case "$CASE" \
  --seed 42 \
  --lr 0.0005 \
  --iterations "${ITERATIONS:-3000}" \
  --display_every "${DISPLAY_EVERY:-500}" \
  --num_domain 1024 \
  --num_test 512 \
  --num_boundary 160 \
  --num_observe 300 \
  --num_val_observe 300 \
  --num_eval_observe 300 \
  --noise_level 0.0 \
  --reg_weight 0.0001 \
  --data_weight 20.0 \
  --boundary_weight 20.0 \
  --lambda_floor 0.1 \
  --mu_floor 0.1 \
  --activation tanh \
  --backbone_type mlp \
  --hidden_layers 128,128,128,128 \
  --state_layers 128,128,128,128 \
  --material_layers 128,128,128,128 \
  --geometry_layers 64,64,64 \
  --interface_layers 128,128,128,128 \
  --interface_sharpness 10.0 \
  --num_frequencies 4 \
  --material_parameterization bulkmu \
  --k_floor 0.2 \
  --load_scales 1.0,1.0,1.0 \
  --load_modes normal_to_layer,bending_y,biaxial_bulk \
  --load_balance_mode reference_rms \
  --load_balance_nx 48 \
  --load_balance_ny 48 \
  --load_balance_epsilon 1e-06 \
  --primary_load_index 0 \
  --observation_split_tag revision_uniform_smoke_v1 \
  --observation_cache_dir "$REPO/exp/_shared_observation_splits/spatial_material_inverse" \
  --observation_split_seed "${OBSERVATION_SPLIT_SEED:-1059}" \
  --reaction_weight 0.0 \
  --reaction_edge top \
  --exp_root "$REPO/exp/09.COMGERevisionSmoke" \
  --experiment_group "${EXPERIMENT_GROUP:-GradientScope}" \
  --run_name "$RUN_NAME" \
  --save_dir "$SAVE_DIR" \
  --eval_nx 81 \
  --eval_ny 81 \
  --run_eval_after_train \
  --dynamic_loss_balance material_gradnorm \
  --dynamic_balance_period 500 \
  --dynamic_balance_parameter_scope "$SCOPE" \
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
  --teacher_source analytic

echo "[end] $(date '+%F %T %Z') status=$?"
