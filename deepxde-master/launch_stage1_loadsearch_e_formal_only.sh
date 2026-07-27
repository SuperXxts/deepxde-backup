#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
WORK=$ROOT/my_example/spatial_material_inverse
FORMAL_GROUP=Stage1-LoadSearch-60k

COMMON_ARGS=(
  --seed 42
  --num_domain 8000
  --num_test 4000
  --num_boundary 800
  --num_observe 1000
  --num_val_observe 400
  --num_eval_observe 1000
  --run_eval_after_train
  --exp_root "$ROOT/exp"
  --material_parameterization bulkmu
  --load_balance_mode reference_rms
  --primary_load_index 0
  --backbone_type mlp
  --hidden_layers 128,128,128,128
  --state_layers 128,128,128,128
  --material_layers 128,128,128,128
  --method fivestate_stress_kmu
  --lr 5e-4
)

FORMAL_ARGS=(
  --iterations 60000
  --display_every 1000
)

build_cmd() {
  printf '%q ' "$@"
}

launch_one() {
  local node=$1
  local dcu=$2
  local case_name=$3
  local obs_tag=$4
  local run_key=$5
  local load_modes=$6
  local load_scales=$7
  local formal_dir=$ROOT/exp/$FORMAL_GROUP/$case_name/${run_key}_60k
  local wrapper=$ROOT/exp/$FORMAL_GROUP/$case_name/${run_key}_launcher.sh
  local screen_name=codex260320_${run_key}
  local formal_cmd

  mkdir -p "$formal_dir" "$ROOT/exp/$FORMAL_GROUP/$case_name"

  formal_cmd=$(build_cmd python train_material_field.py "${COMMON_ARGS[@]}" --case "$case_name" --observation_split_tag "$obs_tag" --load_modes "$load_modes" --load_scales "$load_scales" --experiment_group "$FORMAL_GROUP" --run_name "${run_key}_60k" "${FORMAL_ARGS[@]}")

  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
set -eo pipefail
mkdir -p '${formal_dir}'
exec > '${formal_dir}/screen.log' 2>&1
export PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:\$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:\${LD_LIBRARY_PATH:-}
set +u
source /etc/profile || true
source ~/.bashrc || true
source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh || true
set -u
conda activate pytorch2.1w2
export CUDA_VISIBLE_DEVICES=${dcu}
export ROCR_VISIBLE_DEVICES=${dcu}
export HIP_VISIBLE_DEVICES=${dcu}
cd ${WORK}
${formal_cmd}
EOF
  chmod +x "$wrapper"
  ssh "$node" "screen -ls | grep -q '\\.deepxde' && timeout 5s screen -r -d deepxde >/dev/null 2>&1 || true; screen -S ${screen_name} -X quit >/dev/null 2>&1 || true; screen -dmS ${screen_name} bash ${wrapper}"
  echo "launched ${run_key} on ${node} dcu${dcu}"
}

launch_one 10.6.234.22 0 layered stage1_layered_sparse_v1 layered_l1_normal_shear_bulk normal_to_layer,cross_layer_shear,biaxial_bulk 1.0,1.0,1.0
launch_one 10.6.234.23 0 layered stage1_layered_sparse_v1 layered_l2_normal_shear_nonuniform normal_to_layer,cross_layer_shear,top_nonuniform_compression 1.0,1.0,1.0
launch_one 10.6.234.24 0 layered stage1_layered_sparse_v1 layered_l3_normal_bending_bulk normal_to_layer,bending_y,biaxial_bulk 1.0,1.0,1.0
launch_one 10.6.234.25 0 single_inclusion stage1_single_sparse_v1 single_s1_bulk_shear_ux biaxial_bulk,pure_shear,uniaxial_x 1.0,1.0,1.0
launch_one 10.6.234.27 0 single_inclusion stage1_single_sparse_v1 single_s2_bulk_shear_uy biaxial_bulk,pure_shear,uniaxial_y 1.0,1.0,1.0
launch_one 10.6.234.28 0 single_inclusion stage1_single_sparse_v1 single_s3_bulk_shear_bending biaxial_bulk,pure_shear,bending_y 1.0,1.0,1.0
launch_one 10.6.234.29 0 single_inclusion stage1_single_sparse_v1 single_s4_bulk_shear_nonuniform biaxial_bulk,pure_shear,top_nonuniform_compression 1.0,1.0,1.0
launch_one 10.6.234.30 0 single_inclusion stage1_single_sparse_v1 single_s5_bulk_shear_patch biaxial_bulk,pure_shear,edge_patch_load 1.0,1.0,1.0
