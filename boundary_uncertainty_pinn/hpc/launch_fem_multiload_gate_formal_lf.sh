#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
GROUP=${GROUP:-02.FEMThreeLoadFormal}
ITERATIONS=${ITERATIONS:-150000}
DISPLAY_EVERY=${DISPLAY_EVERY:-1000}
DYNAMIC_FIGURE_EVERY=${DYNAMIC_FIGURE_EVERY:-$DISPLAY_EVERY}
MATERIAL_MONITOR_EVERY=${MATERIAL_MONITOR_EVERY:-$DISPLAY_EVERY}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5000}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}
MATERIAL_SMOOTHNESS_WEIGHT=${MATERIAL_SMOOTHNESS_WEIGHT:-0.0}

CASES=(C1 C2 C3 C4 C5 C6 C7)
RUN_NAMES=(B0 B1 B2 B3 B4 B5 B6)
NODES=(comput1 comput1 comput2 comput2 comput3 comput3 comput4)
DEVICES=(0 1 0 1 0 1 0)
RUN_NOTES=(
  B0_three_load_true_boundary
  B1_three_load_wrong_fixed_boundary
  B2_three_load_learnable_boundary
  B3_three_load_learnable_boundary_reaction
  B4_three_load_learnable_boundary_reaction_anchor
  B5_three_load_boundary_branch_reaction_anchor
  B6_three_load_full_decoupled_method
)

for i in "${!CASES[@]}"; do
  CASE=${CASES[$i]}
  RUN_NAME=${RUN_NAMES[$i]}
  NODE=${NODES[$i]}
  DEVICE_INDEX=${DEVICES[$i]}
  RUN_NOTE=${RUN_NOTES[$i]}
  RUN_DIR="$PROJECT/exp/$GROUP/$RUN_NAME"
  LAUNCH_LOG="$PROJECT/exp/$GROUP/_launch_logs/${RUN_NAME}_${RUN_STAMP}.log"
  SCREEN_NAME="bupinn_${RUN_NAME}_${RUN_STAMP}"
  remote_env="CASE=$CASE RUN_NAME=$RUN_NAME RUN_STAMP=$RUN_STAMP DEVICE_INDEX=$DEVICE_INDEX ITERATIONS=$ITERATIONS DISPLAY_EVERY=$DISPLAY_EVERY DYNAMIC_FIGURE_EVERY=$DYNAMIC_FIGURE_EVERY MATERIAL_MONITOR_EVERY=$MATERIAL_MONITOR_EVERY CHECKPOINT_EVERY=$CHECKPOINT_EVERY SEED=$SEED GROUP=$GROUP CONDA_ENV=$CONDA_ENV MATERIAL_SMOOTHNESS_WEIGHT=$MATERIAL_SMOOTHNESS_WEIGHT RUN_NOTE=$RUN_NOTE"
  echo "launch case=$CASE node=$NODE device=$DEVICE_INDEX run_dir=$RUN_DIR"
  ssh "$NODE" "set -euo pipefail; cd '$PROJECT'; \
    if [ -e '$RUN_DIR' ]; then echo 'ERROR: run directory already exists: $RUN_DIR' >&2; exit 2; fi; \
    chmod +x hpc/run_fem_multiload_gate_onecase_lf.sh; \
    mkdir -p '$(dirname "$LAUNCH_LOG")'; \
    if command -v screen >/dev/null 2>&1; then \
      if screen -ls | grep -q '[.]$SCREEN_NAME[[:space:]]'; then echo 'ERROR: screen already exists: $SCREEN_NAME' >&2; exit 2; fi; \
      screen -dmS '$SCREEN_NAME' bash -lc '$remote_env bash hpc/run_fem_multiload_gate_onecase_lf.sh'; \
      echo 'launcher=screen screen=$SCREEN_NAME run_dir=$RUN_DIR' > '$LAUNCH_LOG'; \
    else \
      nohup bash -lc '$remote_env bash hpc/run_fem_multiload_gate_onecase_lf.sh' > '$LAUNCH_LOG' 2>&1 & \
      echo 'launcher=nohup pid='\$!' run_dir=$RUN_DIR' >> '$LAUNCH_LOG'; \
    fi"
done

echo "run_stamp=$RUN_STAMP"
echo "group=$GROUP"
echo "runs=B0 B1 B2 B3 B4 B5 B6"
echo "case_mapping=B0:C1 B1:C2 B2:C3 B3:C4 B4:C5 B5:C6 B6:C7"
echo "display_every=$DISPLAY_EVERY"
echo "dynamic_figure_every=$DYNAMIC_FIGURE_EVERY"
echo "material_monitor_every=$MATERIAL_MONITOR_EVERY"
echo "material_smoothness_weight=$MATERIAL_SMOOTHNESS_WEIGHT"
echo "eta=about 18-22 hours; update after the first live 1000-step interval"
