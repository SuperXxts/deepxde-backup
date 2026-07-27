#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$ROOT/my_example/spatial_material_inverse
CASE_NAME=single_inclusion
SMOKE_GROUP=StructDiag-KMu-Smoke2k
FORMAL_GROUP=StructDiag-KMu-60k
LOAD_MODES=biaxial_bulk,uniaxial_x,pure_shear
LOAD_SCALES=1.0,1.0,1.0
OBS_TAG=kmu_structdiag_v2

COMMON_DATA_ARGS=(
  --case "$CASE_NAME"
  --seed 42
  --display_every 1000
  --num_domain 8000
  --num_test 4000
  --num_boundary 800
  --num_observe 1000
  --num_val_observe 400
  --num_eval_observe 1000
  --observation_split_tag "$OBS_TAG"
  --run_eval_after_train
  --exp_root "$ROOT/exp"
  --material_parameterization bulkmu
  --load_scales "$LOAD_SCALES"
  --load_modes "$LOAD_MODES"
  --primary_load_index 0
  --backbone_type mlp
  --hidden_layers 128,128,128,128
  --state_layers 128,128,128,128
  --material_layers 128,128,128,128
)

DIRECT_SMOKE_COMMON=(
  --iterations 2000
  --display_every 200
  --lr 5e-4
)
DIRECT_FORMAL_COMMON=(
  --iterations 60000
  --display_every 1000
  --lr 5e-4
)

PINN_SMOKE=("${DIRECT_SMOKE_COMMON[@]}" --method pinn)
PINN_FORMAL=("${DIRECT_FORMAL_COMMON[@]}" --method pinn)
TWOBRANCH_COMPACT_SMOKE=("${DIRECT_SMOKE_COMMON[@]}" --method twobranch_compact_kmu)
TWOBRANCH_COMPACT_FORMAL=("${DIRECT_FORMAL_COMMON[@]}" --method twobranch_compact_kmu)
TWOBRANCH_STRESS_SMOKE=("${DIRECT_SMOKE_COMMON[@]}" --method twobranch_stress_kmu)
TWOBRANCH_STRESS_FORMAL=("${DIRECT_FORMAL_COMMON[@]}" --method twobranch_stress_kmu)
PFNN_SCALAR_SMOKE=("${DIRECT_SMOKE_COMMON[@]}" --method pfnn_scalar_kmu --activation tanh)
PFNN_SCALAR_FORMAL=("${DIRECT_FORMAL_COMMON[@]}" --method pfnn_scalar_kmu --activation tanh)
FIVESTATE_STRESS_SMOKE=("${DIRECT_SMOKE_COMMON[@]}" --method fivestate_stress_kmu)
FIVESTATE_STRESS_FORMAL=("${DIRECT_FORMAL_COMMON[@]}" --method fivestate_stress_kmu)

build_cmd() {
  printf '%q ' "$@"
}

launch_one() {
  local node=$1
  local dcu=$2
  local run_key=$3
  local smoke_name=$4
  local formal_name=$5
  local -n smoke_args=$smoke_name
  local -n formal_args=$formal_name
  local smoke_dir=$ROOT/exp/$SMOKE_GROUP/$CASE_NAME/${run_key}_smoke2k
  local formal_dir=$ROOT/exp/$FORMAL_GROUP/$CASE_NAME/${run_key}_60k
  local wrapper=$ROOT/exp/$FORMAL_GROUP/$CASE_NAME/${run_key}_launcher.sh
  local screen_name=codex260320_${run_key}
  local smoke_cmd formal_cmd

  mkdir -p "$smoke_dir" "$formal_dir"
  smoke_cmd=$(build_cmd python train_material_field.py "${COMMON_DATA_ARGS[@]}" --experiment_group "$SMOKE_GROUP" --run_name "${run_key}_smoke2k" "${smoke_args[@]}")
  formal_cmd=$(build_cmd python train_material_field.py "${COMMON_DATA_ARGS[@]}" --experiment_group "$FORMAL_GROUP" --run_name "${run_key}_60k" "${formal_args[@]}")

  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
run_stage() {
  local log_file=\$1
  shift
  (
    set -eo pipefail
    exec > "\$log_file" 2>&1
    set +u
    source /etc/profile || true
    source ~/.bashrc || true
    set -u
    conda activate pytorch2.1w2
    export CUDA_VISIBLE_DEVICES=${dcu}
    cd ${WORK}
    "\$@"
  )
}
run_stage '${smoke_dir}/screen.log' bash -c ${smoke_cmd@Q}
run_stage '${formal_dir}/screen.log' bash -c ${formal_cmd@Q}
EOF
  chmod +x "$wrapper"
  ssh "$node" "screen -dmS ${screen_name} bash ${wrapper}"
  echo "launched ${run_key} on ${node} dcu${dcu}"
}

mkdir -p "$ROOT/exp/$SMOKE_GROUP/$CASE_NAME" "$ROOT/exp/$FORMAL_GROUP/$CASE_NAME"

launch_one 10.6.234.22 0 pinn_mlp_kmu_3load PINN_SMOKE PINN_FORMAL
launch_one 10.6.234.23 0 twobranch_compact_mlp_kmu_3load TWOBRANCH_COMPACT_SMOKE TWOBRANCH_COMPACT_FORMAL
launch_one 10.6.234.24 0 twobranch_stress_mlp_kmu_3load TWOBRANCH_STRESS_SMOKE TWOBRANCH_STRESS_FORMAL
launch_one 10.6.234.25 0 pfnn_scalar_kmu_3load PFNN_SCALAR_SMOKE PFNN_SCALAR_FORMAL
launch_one 10.6.234.27 0 fivestate_stress_mlp_kmu_3load FIVESTATE_STRESS_SMOKE FIVESTATE_STRESS_FORMAL
