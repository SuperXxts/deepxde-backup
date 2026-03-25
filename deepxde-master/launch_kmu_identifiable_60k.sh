#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$ROOT/my_example/spatial_material_inverse
CASE_NAME=single_inclusion
EXP_GROUP=KMu-Identifiable-Benchmark-60k
LOAD_MODES=biaxial_bulk,uniaxial_x,pure_shear
LOAD_SCALES=1.0,1.0,1.0
COMMON_ARGS=--case ${CASE_NAME} --seed 42 --iterations 60000 --display_every 1000 --num_domain 8000 --num_test 4000 --num_boundary 800 --num_observe 1000 --num_val_observe 400 --num_eval_observe 1000 --observation_split_tag kmu_identifiable_v1 --run_eval_after_train --exp_root $ROOT/exp --experiment_group $EXP_GROUP --material_parameterization bulkmu --staged_training --warmup_iterations 1200 --geometry_stage_iterations 2400 --material_stage_iterations 1200 --geometry_stage_physics_scale 0.02 --material_stage_physics_scale 0.05 --adaptive_main_stage --adaptive_main_chunks 4 --adaptive_main_strategy grad_balance --adaptive_main_base_physics_scale 0.35 --adaptive_main_reg_scale 1.0 --adaptive_main_data_scale 1.5 --adaptive_main_boundary_scale 1.5 --adaptive_main_signal_points 512 --adaptive_main_signal_observe_points 256 --adaptive_main_signal_boundary_points 256 --adaptive_main_min_physics_scale 0.1 --adaptive_main_max_physics_scale 1.0 --adaptive_main_min_data_scale 0.5 --adaptive_main_max_data_scale 2.5 --adaptive_main_min_boundary_scale 0.5 --adaptive_main_max_boundary_scale 2.5 --adaptive_main_scale_ema 0.5 --main_stage_lr_start 5e-4 --main_stage_lr_end 5e-5 --state_layers 128,128,128,128 --geometry_layers 64,64 --interface_sharpness 12 --load_scales ${LOAD_SCALES} --load_modes ${LOAD_MODES} --primary_load_index 0

launch_one() {
  local node=$1
  local dcu=$2
  local method=$3
  local run_name=$4
  local extra_args=$5
  local run_dir=$ROOT/exp/${EXP_GROUP}/${CASE_NAME}/${run_name}
  local screen_name=codex260320_${run_name}
  local cmd=source /public/home/xinxi/wxtian/anaconda3/bin/activate pytorch2.1w2 && export CUDA_VISIBLE_DEVICES=${dcu} && cd ${WORK} && python train_material_field.py --method ${method} ${COMMON_ARGS} --run_name ${run_name} ${extra_args}
  ssh $node mkdir -p '${run_dir}' && screen -dmS '${screen_name}' bash -lc "${cmd} > '${run_dir}/screen.log' 2>&1"
  echo launched ${run_name} on ${node} dcu${dcu}
}

launch_one 10.6.234.29 1 pinn pinn_kmu_3load_60k --hidden_layers 128,128,128,128
launch_one 10.6.234.30 1 geoiaminn_v3 geoiaminnv3_kmu_3load_60k "
