#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$ROOT/my_example/spatial_material_inverse
CASE_NAME=single_inclusion
SMOKE_GROUP=KMu-StrategySearch-Smoke2k
FORMAL_GROUP=KMu-StrategySearch-60k
LOAD_MODES=biaxial_bulk,uniaxial_x,pure_shear
LOAD_SCALES=1.0,1.0,1.0
OBS_TAG=kmu_identifiable_v1

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
)

PINN_SMOKE_ARGS=(
  --method pinn
  --iterations 2000
  --display_every 200
  --staged_training
  --warmup_iterations 400
  --main_lr 5e-4
  --main_stage_chunks 4
  --main_stage_physics_scale_start 0.1
  --main_stage_physics_scale_end 0.6
  --main_stage_reg_scale_start 1.0
  --main_stage_reg_scale_end 1.0
  --main_stage_data_scale_start 2.0
  --main_stage_data_scale_end 1.0
  --main_stage_boundary_scale_start 2.0
  --main_stage_boundary_scale_end 1.0
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 1e-4
  --hidden_layers 128,128,128,128
)
PINN_FORMAL_ARGS=(
  --method pinn
  --iterations 60000
  --staged_training
  --warmup_iterations 1200
  --main_lr 5e-4
  --main_stage_chunks 16
  --main_stage_physics_scale_start 0.1
  --main_stage_physics_scale_end 0.6
  --main_stage_reg_scale_start 1.0
  --main_stage_reg_scale_end 1.0
  --main_stage_data_scale_start 2.0
  --main_stage_data_scale_end 1.0
  --main_stage_boundary_scale_start 2.0
  --main_stage_boundary_scale_end 1.0
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 5e-5
  --hidden_layers 128,128,128,128
)
PINN_RESMLP_SMOKE_ARGS=("${PINN_SMOKE_ARGS[@]}" --backbone_type resmlp)
PINN_RESMLP_FORMAL_ARGS=("${PINN_FORMAL_ARGS[@]}" --backbone_type resmlp)

GEO_SMOKE_COMMON=(
  --method geoiaminn_v3
  --iterations 2000
  --display_every 200
  --staged_training
  --warmup_iterations 400
  --geometry_stage_iterations 600
  --material_stage_iterations 400
  --geometry_stage_physics_scale 0.02
  --material_stage_physics_scale 0.05
  --adaptive_main_stage
  --adaptive_main_chunks 4
  --adaptive_main_base_physics_scale 0.35
  --adaptive_main_reg_scale 1.0
  --adaptive_main_data_scale 1.5
  --adaptive_main_boundary_scale 1.5
  --adaptive_main_min_physics_scale 0.1
  --adaptive_main_max_physics_scale 1.0
  --adaptive_main_min_data_scale 0.5
  --adaptive_main_max_data_scale 2.5
  --adaptive_main_min_boundary_scale 0.5
  --adaptive_main_max_boundary_scale 2.5
  --adaptive_main_scale_ema 0.5
  --main_lr 5e-4
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 1e-4
  --state_layers 128,128,128,128
  --geometry_layers 64,64
  --interface_sharpness 12
)
GEO_FORMAL_COMMON=(
  --method geoiaminn_v3
  --iterations 60000
  --staged_training
  --warmup_iterations 1200
  --geometry_stage_iterations 2400
  --material_stage_iterations 1200
  --geometry_stage_physics_scale 0.02
  --material_stage_physics_scale 0.05
  --adaptive_main_stage
  --adaptive_main_chunks 4
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
  --main_lr 5e-4
  --main_stage_lr_start 5e-4
  --main_stage_lr_end 5e-5
  --state_layers 128,128,128,128
  --geometry_layers 64,64
  --interface_sharpness 12
)

GEO_GRADBAL_SMOKE=("${GEO_SMOKE_COMMON[@]}" --adaptive_main_strategy grad_balance)
GEO_GRADBAL_FORMAL=("${GEO_FORMAL_COMMON[@]}" --adaptive_main_strategy grad_balance)
GEO_MULTILOSS_SMOKE=("${GEO_SMOKE_COMMON[@]}" --adaptive_main_strategy multiloss)
GEO_MULTILOSS_FORMAL=("${GEO_FORMAL_COMMON[@]}" --adaptive_main_strategy multiloss)
GEO_VALFB_SMOKE=("${GEO_SMOKE_COMMON[@]}" --adaptive_main_strategy val_feedback)
GEO_VALFB_FORMAL=("${GEO_FORMAL_COMMON[@]}" --adaptive_main_strategy val_feedback)
GEO_PHYSSTRONG_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --geometry_stage_physics_scale 0.05 --material_stage_physics_scale 0.10 --adaptive_main_base_physics_scale 0.50 --adaptive_main_max_physics_scale 1.50)
GEO_PHYSSTRONG_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --geometry_stage_physics_scale 0.05 --material_stage_physics_scale 0.10 --adaptive_main_base_physics_scale 0.50 --adaptive_main_max_physics_scale 1.50)
GEO_PHYSSOFT_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --geometry_stage_physics_scale 0.01 --material_stage_physics_scale 0.03 --adaptive_main_base_physics_scale 0.25 --adaptive_main_max_physics_scale 0.80)
GEO_PHYSSOFT_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --geometry_stage_physics_scale 0.01 --material_stage_physics_scale 0.03 --adaptive_main_base_physics_scale 0.25 --adaptive_main_max_physics_scale 0.80)
GEO_GEOMWIDE_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --geometry_layers 128,128,128)
GEO_GEOMWIDE_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --geometry_layers 128,128,128)
GEO_GEOMDEEP_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --geometry_layers 128,128,128,128)
GEO_GEOMDEEP_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --geometry_layers 128,128,128,128)
GEO_BULKCONTRAST_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)
GEO_BULKCONTRAST_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)
GEO_BINARY_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --geometry_stage_binary_weight 0.05 --geometry_stage_interface_sharpness 16)
GEO_BINARY_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --geometry_stage_binary_weight 0.05 --geometry_stage_interface_sharpness 16)
GEO_RESMLP_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --backbone_type resmlp)
GEO_RESMLP_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --backbone_type resmlp)
GEO_SHARP8_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --interface_sharpness 8)
GEO_SHARP8_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --interface_sharpness 8)
GEO_SHARP16_SMOKE=("${GEO_GRADBAL_SMOKE[@]}" --interface_sharpness 16)
GEO_SHARP16_FORMAL=("${GEO_GRADBAL_FORMAL[@]}" --interface_sharpness 16)
GEO_PHYSSTRONG_BULKCONTRAST_SMOKE=("${GEO_PHYSSTRONG_SMOKE[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)
GEO_PHYSSTRONG_BULKCONTRAST_FORMAL=("${GEO_PHYSSTRONG_FORMAL[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)
GEO_GEOMWIDE_BULKCONTRAST_SMOKE=("${GEO_GEOMWIDE_SMOKE[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)
GEO_GEOMWIDE_BULKCONTRAST_FORMAL=("${GEO_GEOMWIDE_FORMAL[@]}" --material_stage_contrast_weight 0.50 --material_stage_bulk_gap_target 0.40 --material_stage_mu_gap_target 0.20)

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

launch_one 10.6.234.22 0 pinn_kmu_sched PINN_SMOKE_ARGS PINN_FORMAL_ARGS
launch_one 10.6.234.22 1 pinn_kmu_resmlp PINN_RESMLP_SMOKE_ARGS PINN_RESMLP_FORMAL_ARGS
launch_one 10.6.234.23 0 geo_kmu_gradbal GEO_GRADBAL_SMOKE GEO_GRADBAL_FORMAL
launch_one 10.6.234.23 1 geo_kmu_multiloss GEO_MULTILOSS_SMOKE GEO_MULTILOSS_FORMAL
launch_one 10.6.234.24 0 geo_kmu_valfb GEO_VALFB_SMOKE GEO_VALFB_FORMAL
launch_one 10.6.234.24 1 geo_kmu_physstrong GEO_PHYSSTRONG_SMOKE GEO_PHYSSTRONG_FORMAL
launch_one 10.6.234.25 0 geo_kmu_physsoft GEO_PHYSSOFT_SMOKE GEO_PHYSSOFT_FORMAL
launch_one 10.6.234.25 1 geo_kmu_geomwide GEO_GEOMWIDE_SMOKE GEO_GEOMWIDE_FORMAL
launch_one 10.6.234.27 0 geo_kmu_geomdeep GEO_GEOMDEEP_SMOKE GEO_GEOMDEEP_FORMAL
launch_one 10.6.234.27 1 geo_kmu_bulkcontrast GEO_BULKCONTRAST_SMOKE GEO_BULKCONTRAST_FORMAL
launch_one 10.6.234.28 0 geo_kmu_binary GEO_BINARY_SMOKE GEO_BINARY_FORMAL
launch_one 10.6.234.28 1 geo_kmu_resmlp GEO_RESMLP_SMOKE GEO_RESMLP_FORMAL
launch_one 10.6.234.29 0 geo_kmu_sharp8 GEO_SHARP8_SMOKE GEO_SHARP8_FORMAL
launch_one 10.6.234.29 1 geo_kmu_sharp16 GEO_SHARP16_SMOKE GEO_SHARP16_FORMAL
launch_one 10.6.234.30 0 geo_kmu_physstrong_bulkcontrast GEO_PHYSSTRONG_BULKCONTRAST_SMOKE GEO_PHYSSTRONG_BULKCONTRAST_FORMAL
launch_one 10.6.234.30 1 geo_kmu_geomwide_bulkcontrast GEO_GEOMWIDE_BULKCONTRAST_SMOKE GEO_GEOMWIDE_BULKCONTRAST_FORMAL
