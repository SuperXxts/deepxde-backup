#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master"
WORK="$ROOT/my_example/spatial_material_inverse"
EXP_GROUP="RespSplit-Control-60k"
CASE_NAME="single_inclusion"

COMMON_ARGS="--method geoiaminn_v3 --case ${CASE_NAME} --seed 42 --iterations 60000 --display_every 1000 --num_domain 8000 --num_test 4000 --num_boundary 800 --num_observe 1000 --num_val_observe 400 --num_eval_observe 1000 --observation_split_tag loadaug_dblpts_shared_v1 --run_eval_after_train --exp_root $ROOT/exp --experiment_group $EXP_GROUP --material_parameterization lamemu --staged_training --warmup_iterations 1200 --geometry_stage_iterations 2400 --material_stage_iterations 1200 --geometry_stage_physics_scale 0.10 --material_stage_physics_scale 0.10 --adaptive_main_stage --adaptive_main_chunks 16 --adaptive_main_strategy grad_balance --adaptive_main_base_physics_scale 0.35 --adaptive_main_reg_scale 1.0 --adaptive_main_data_scale 1.5 --adaptive_main_boundary_scale 1.5 --adaptive_main_min_physics_scale 0.1 --adaptive_main_max_physics_scale 1.0 --adaptive_main_min_data_scale 0.5 --adaptive_main_max_data_scale 2.5 --adaptive_main_min_boundary_scale 0.5 --adaptive_main_max_boundary_scale 2.5 --adaptive_main_scale_ema 0.5 --main_stage_lr_start 5e-4 --main_stage_lr_end 5e-5 --state_layers 128,128,128,128 --geometry_layers 64,64 --interface_sharpness 12 --load_scales 1.0,1.0,1.0 --load_modes x_tension,y_tension,shear"

RESP_FREEZE="--freeze_state_geometry_stage --freeze_region_geometry_stage --freeze_state_material_stage --freeze_geometry_material_stage"
RESP_BIN="--geometry_stage_binary_weight 0.10 --material_stage_binary_weight 0.10"
RESP_CONTRAST="--material_stage_contrast_weight 0.50 --material_stage_lambda_gap_target 0.40 --material_stage_mu_gap_target 0.20"
RESP_LOCK4="--main_stage_freeze_geometry_chunks 4"
RESP_LOCK8="--main_stage_freeze_geometry_chunks 8"
RESP_LONG="--geometry_stage_iterations 4800 --material_stage_iterations 2400"
RESP_SHARP="--geometry_stage_interface_sharpness 6 --material_stage_interface_sharpness 10 --main_stage_interface_sharpness 14"

launch_one() {
  local node="$1"
  local dcu="$2"
  local run_name="$3"
  local extra_args="$4"
  local run_dir="$ROOT/exp/${EXP_GROUP}/${CASE_NAME}/${run_name}"
  local screen_name="codex260320_${run_name}"
  local cmd="source /public/home/xinxi/wxtian/anaconda3/bin/activate pytorch2.1w2 && export CUDA_VISIBLE_DEVICES=${dcu} && echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES && cd ${WORK} && python train_material_field.py ${COMMON_ARGS} --run_name ${run_name} ${extra_args}"
  ssh "$node" "mkdir -p '${run_dir}' && screen -dmS '${screen_name}' bash -lc \"${cmd} > '${run_dir}/screen.log' 2>&1\""
  echo "launched ${run_name} on ${node} dcu${dcu}"
}

launch_one 10.6.234.22 0 rs0_stagephys_only_60k ""
launch_one 10.6.234.23 0 rs1_split_freeze_60k "${RESP_FREEZE}"
launch_one 10.6.234.24 0 rs2_split_bin_60k "${RESP_FREEZE} ${RESP_BIN}"
launch_one 10.6.234.25 0 rs3_split_bin_contrast_60k "${RESP_FREEZE} ${RESP_BIN} ${RESP_CONTRAST}"
launch_one 10.6.234.27 0 rs4_split_bin_contrast_lock4_60k "${RESP_FREEZE} ${RESP_BIN} ${RESP_CONTRAST} ${RESP_LOCK4}"
launch_one 10.6.234.28 0 rs5_split_bin_contrast_lock8_60k "${RESP_FREEZE} ${RESP_BIN} ${RESP_CONTRAST} ${RESP_LOCK8}"
launch_one 10.6.234.29 0 rs6_split_bin_contrast_lock8_longgeom_60k "${RESP_FREEZE} ${RESP_BIN} ${RESP_CONTRAST} ${RESP_LOCK8} ${RESP_LONG}"
launch_one 10.6.234.30 0 rs7_split_bin_contrast_lock8_sharp_60k "${RESP_FREEZE} ${RESP_BIN} ${RESP_CONTRAST} ${RESP_LOCK8} ${RESP_SHARP}"
