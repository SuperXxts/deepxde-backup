#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$ROOT/my_example/spatial_material_inverse
CASE_NAME=single_inclusion
SMOKE_GROUP=StructDiag-KMu-Smoke2k
FORMAL_GROUP=StructDiag-KMu-60k
LOAD_MODES=biaxial_bulk,uniaxial_x,pure_shear
LOAD_SCALES=1.0,1.0,1.0
OBS_TAG=kmu_structdiag_v1

COMMON_ARGS=(
  --case "$CASE_NAME"
  --seed 42
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
  --backbone_type resmlp
  --hidden_layers 128,128,128,128
  --state_layers 128,128,128,128
  --material_layers 128,128,128,128
  --geometry_layers 64,64
)

DIRECT_SMOKE=(
  --iterations 2000
  --display_every 200
  --lr 5e-4
)
DIRECT_FORMAL=(
  --iterations 60000
  --display_every 1000
  --lr 5e-4
)

GEO_SMOKE=(
  --method geoiaminn_v3
  --iterations 2000
  --display_every 200
  --staged_training
  --warmup_iterations 400
  --geometry_stage_iterations 600
  --material_stage_iterations 400
  --geometry_stage_physics_scale 0.02
  --material_stage_physics_scale 0.05
  --geometry_stage_binary_weight 0.05
  --geometry_stage_interface_sharpness 16
  --adaptive_main_stage
  --adaptive_main_chunks 4
  --adaptive_main_strategy grad_balance
  --adaptive_main_base_physics_scale 0.35
  --adaptive_main_reg_scale 1.0
  --adaptive_main_data_scale 1.5
  --adaptive_main_boundary_scale 1.5
  --adaptive_main_signal_points 256
  --adaptive_main_signal_observe_points 128
  --adaptive_main_signal_boundary_points 128
  --adaptive_main_min_physics_scale 0.1
  --adaptive_main_max_physics_scale 1.0
  --adaptive_main_min_data_scale 0.5
  --adaptive_main_max_data_scale 2.5
  --adaptive_main_min_boundary_scale 0.5
  --adaptive_main_max_boundary_scale 2.5
  --adaptive_main_scale_ema 0.5
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 1e-4
  --interface_sharpness 12
)
GEO_FORMAL=(
  --method geoiaminn_v3
  --iterations 60000
  --display_every 1000
  --staged_training
  --warmup_iterations 1200
  --geometry_stage_iterations 2400
  --material_stage_iterations 1200
  --geometry_stage_physics_scale 0.02
  --material_stage_physics_scale 0.05
  --geometry_stage_binary_weight 0.05
  --geometry_stage_interface_sharpness 16
  --adaptive_main_stage
  --adaptive_main_chunks 4
  --adaptive_main_strategy grad_balance
  --adaptive_main_base_physics_scale 0.35
  --adaptive_main_reg_scale 1.0
  --adaptive_main_data_scale 1.5
  --adaptive_main_boundary_scale 1.5
  --adaptive_main_signal_points 512
  --adaptive_main_signal_observe_points 256
  --adaptive_main_signal_boundary_points 256
  --adaptive_main_min_physics_scale 0.1
  --adaptive_main_max_physics_scale 1.0
  --adaptive_main_min_data_scale 0.5
  --adaptive_main_max_data_scale 2.5
  --adaptive_main_min_boundary_scale 0.5
  --adaptive_main_max_boundary_scale 2.5
  --adaptive_main_scale_ema 0.5
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 5e-5
  --interface_sharpness 12
)

build_cmd() {
  printf '%q ' "$@"
}

launch_one() {
  local node=$1
  local dcu=$2
  local run_key=$3
  local smoke_extra=$4
  local formal_extra=$5
  local smoke_dir=$ROOT/exp/$SMOKE_GROUP/$CASE_NAME/${run_key}_smoke2k
  local formal_dir=$ROOT/exp/$FORMAL_GROUP/$CASE_NAME/${run_key}_60k
  local wrapper=$ROOT/exp/$FORMAL_GROUP/$CASE_NAME/${run_key}_launcher.sh
  local screen_name=codex260320_${run_key}
  local smoke_cmd formal_cmd

  mkdir -p "$smoke_dir" "$formal_dir"
  smoke_cmd=$(build_cmd python train_material_field.py "${COMMON_ARGS[@]}" --experiment_group "$SMOKE_GROUP" --run_name "${run_key}_smoke2k" $smoke_extra)
  formal_cmd=$(build_cmd python train_material_field.py "${COMMON_ARGS[@]}" --experiment_group "$FORMAL_GROUP" --run_name "${run_key}_60k" $formal_extra)

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

launch_one 10.6.234.22 0 pinn_resmlp_kmu_3load "--method pinn ${DIRECT_SMOKE[*]}" "--method pinn ${DIRECT_FORMAL[*]}"
launch_one 10.6.234.23 0 twobranch_compact_kmu_3load "--method twobranch_compact_kmu ${DIRECT_SMOKE[*]}" "--method twobranch_compact_kmu ${DIRECT_FORMAL[*]}"
launch_one 10.6.234.24 0 twobranch_stress_kmu_3load "--method twobranch_stress_kmu ${DIRECT_SMOKE[*]}" "--method twobranch_stress_kmu ${DIRECT_FORMAL[*]}"
launch_one 10.6.234.25 0 fivestate_stress_kmu_3load "--method fivestate_stress_kmu ${DIRECT_SMOKE[*]}" "--method fivestate_stress_kmu ${DIRECT_FORMAL[*]}"
launch_one 10.6.234.27 0 geoiaminnv3_binary_resmlp_kmu_3load "${GEO_SMOKE[*]}" "${GEO_FORMAL[*]}"
