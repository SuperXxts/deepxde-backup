#!/usr/bin/env bash
set -euo pipefail
REPO=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
GROUP=Stage2-Balanced1200-GradNorm-TwoBranch-100k
EXP_ROOT="$REPO/exp"
COMMON_ARGS=(
  --method twobranch_stress_kmu
  --seed 42
  --lr 0.0005
  --iterations 100000
  --display_every 1000
  --num_domain 8000
  --num_test 4000
  --num_boundary 800
  --num_observe 1200
  --num_val_observe 1200
  --num_eval_observe 1200
  --noise_level 0.0
  --reg_weight 0.0001
  --data_weight 20.0
  --boundary_weight 20.0
  --lambda_floor 0.1
  --mu_floor 0.1
  --activation tanh
  --backbone_type mlp
  --hidden_layers 128,128,128,128
  --state_layers 128,128,128,128
  --material_layers 128,128,128,128
  --geometry_layers 64,64,64
  --interface_layers 128,128,128,128
  --interface_sharpness 10.0
  --num_frequencies 4
  --material_parameterization bulkmu
  --k_floor 0.2
  --load_scales 1.0,1.0,1.0
  --load_balance_mode reference_rms
  --load_balance_nx 48
  --load_balance_ny 48
  --load_balance_epsilon 1e-06
  --primary_load_index 0
  --exp_root "$EXP_ROOT"
  --eval_nx 141
  --eval_ny 141
  --device_index 0
  --run_eval_after_train
  --warmup_iterations 1000
  --warmup_lr 0.001
  --main_lr 0.0005
  --warmup_physics_scale 0.0
  --warmup_reg_scale 0.0
  --geometry_stage_iterations 0
  --geometry_stage_lr 0.0008
  --geometry_stage_physics_scale 0.02
  --geometry_stage_reg_scale 0.0
  --geometry_stage_data_scale 1.0
  --geometry_stage_boundary_scale 1.0
  --geometry_stage_balance_target 0.5
  --geometry_stage_balance_weight 0.0
  --geometry_stage_binary_weight 0.0
  --geometry_stage_interface_sharpness -1.0
  --material_stage_iterations 0
  --material_stage_lr 0.0008
  --material_stage_physics_scale 0.05
  --material_stage_reg_scale 1.0
  --material_stage_usage_floor 0.05
  --material_stage_usage_weight 20.0
  --material_stage_binary_weight 0.05
  --material_stage_contrast_weight 0.0
  --material_stage_lambda_gap_target 0.0
  --material_stage_bulk_gap_target 0.0
  --material_stage_mu_gap_target 0.0
  --geometry_prior_weight 0.0
  --layer_y_prior_target 0.5
  --layer_y_init -1.0
  --main_stage_freeze_geometry_chunks 0
  --material_stage_interface_sharpness -1.0
  --main_stage_interface_sharpness -1.0
  --refinement_stage_iterations 0
  --refinement_stage_lr 0.0002
  --refinement_stage_physics_scale 1.0
  --refinement_stage_reg_scale 1.0
  --refinement_stage_data_scale 1.0
  --refinement_stage_boundary_scale 1.0
  --refinement_stage_interface_sharpness -1.0
  --adaptive_main_chunks 4
  --adaptive_main_base_physics_scale 1.0
  --adaptive_main_min_physics_scale 0.25
  --adaptive_main_max_physics_scale 4.0
  --adaptive_main_scale_up 1.35
  --adaptive_main_scale_down 0.75
  --adaptive_main_obs_guard 1.15
  --adaptive_main_reg_scale 1.0
  --adaptive_main_data_scale 1.0
  --adaptive_main_boundary_scale 1.0
  --adaptive_main_domain_points 2048
  --adaptive_main_signal_points 512
  --adaptive_main_signal_observe_points 256
  --adaptive_main_signal_boundary_points 256
  --adaptive_main_strategy val_feedback
  --adaptive_main_scale_ema 0.5
  --adaptive_main_min_data_scale 0.25
  --adaptive_main_max_data_scale 4.0
  --adaptive_main_min_boundary_scale 0.25
  --adaptive_main_max_boundary_scale 4.0
  --adaptive_main_grad_eps 1e-12
  --main_stage_chunks 1
  --main_stage_physics_scale_start 1.0
  --main_stage_physics_scale_end 1.0
  --main_stage_reg_scale_start 1.0
  --main_stage_reg_scale_end 1.0
  --main_stage_data_scale_start 1.0
  --main_stage_data_scale_end 1.0
  --main_stage_boundary_scale_start 1.0
  --main_stage_boundary_scale_end 1.0
  --main_stage_lr_start -1.0
  --main_stage_lr_end -1.0
  --geometry_stage_domain_points 1024
  --material_stage_domain_points 1024
  --dynamic_loss_balance material_gradnorm
  --dynamic_balance_period 1000
  --dynamic_balance_ema 0.5
  --dynamic_balance_reg_scale 1.0
  --dynamic_balance_min_physics_scale 0.25
  --dynamic_balance_max_physics_scale 4.0
  --dynamic_balance_min_observation_scale 0.25
  --dynamic_balance_max_observation_scale 4.0
  --dynamic_balance_min_boundary_scale 0.25
  --dynamic_balance_max_boundary_scale 4.0
  --dynamic_balance_grad_eps 1e-12
)
launch_run() {
  local node="$1" case_name="$2" run_name="$3" load_modes="$4" split_tag="$5"
  local save_dir="$EXP_ROOT/$GROUP/$case_name/$run_name"
  local wrapper="$save_dir/${run_name}_launcher.sh"
  mkdir -p "$save_dir" "$save_dir/png" "$save_dir/json" "$save_dir/metrics" "$save_dir/model" "$save_dir/txt" "$save_dir/npz"
  cat > "$wrapper" <<WRAP
#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:\$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:\${LD_LIBRARY_PATH:-}
source /etc/profile || true
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh || true
conda activate pytorch2.1w2
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
cd "$REPO"
python -u my_example/spatial_material_inverse/train_material_field.py \
  "\${COMMON_ARGS[@]}" \
  --case "$case_name" \
  --load_modes "$load_modes" \
  --observation_split_tag "$split_tag" \
  --experiment_group "$GROUP" \
  --run_name "$run_name" \
  --save_dir "$save_dir" \
  >> "$save_dir/screen.log" 2>&1
WRAP
  chmod +x "$wrapper"
  ssh wxtian@"$node" "screen -r -d deepxde >/dev/null 2>&1 || true; screen -S codex_${run_name} -X quit >/dev/null 2>&1 || true; screen -dmS codex_${run_name} bash '$wrapper'"
}
export -f launch_run

launch_run 10.6.234.22 layered layered_l1_normal_shear_bulk_twobranch_gradnorm_100k normal_to_layer,cross_layer_shear,biaxial_bulk stage2_layered_balanced1200_v1
launch_run 10.6.234.23 layered layered_l2_normal_shear_nonuniform_twobranch_gradnorm_100k normal_to_layer,cross_layer_shear,top_nonuniform_compression stage2_layered_balanced1200_v1
launch_run 10.6.234.24 layered layered_l3_normal_bending_bulk_twobranch_gradnorm_100k normal_to_layer,bending_y,biaxial_bulk stage2_layered_balanced1200_v1
launch_run 10.6.234.25 single_inclusion single_s1_bulk_shear_ux_twobranch_gradnorm_100k biaxial_bulk,pure_shear,uniaxial_x stage2_single_balanced1200_v1
launch_run 10.6.234.27 single_inclusion single_s2_bulk_shear_uy_twobranch_gradnorm_100k biaxial_bulk,pure_shear,uniaxial_y stage2_single_balanced1200_v1
launch_run 10.6.234.28 single_inclusion single_s4_bulk_shear_nonuniform_twobranch_gradnorm_100k biaxial_bulk,pure_shear,top_nonuniform_compression stage2_single_balanced1200_v1
