#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
EXP="$PROJECT/exp"
STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$EXP/_archive_old_exp_$STAMP"

case "$EXP" in
  /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn/exp) ;;
  *)
    echo "ERROR: unsafe exp path: $EXP" >&2
    exit 2
    ;;
esac

if ps -u "${USER:-wxtian}" -f | grep boundary_uncertainty_pinn | grep train_deepxde_pfnn.py | grep -v grep; then
  echo "ERROR: active boundary_uncertainty_pinn training process exists; refuse to archive." >&2
  exit 2
fi

mkdir -p "$ARCHIVE"

for item in \
  "$EXP/00.Smoke" \
  "$EXP/01.AnalyticalBenchmark" \
  "$EXP/_archive_torch_prototype"
do
  if [ -e "$item" ]; then
    echo "ARCHIVE $item -> $ARCHIVE/"
    mv "$item" "$ARCHIVE/"
  fi
done

mkdir -p \
  "$EXP/00.Smoke" \
  "$EXP/01.AnalyticalBenchmark" \
  "$EXP/02.FEMBenchmark" \
  "$EXP/03.NoiseRobustness" \
  "$EXP/04.ObservationSparsity" \
  "$EXP/05.AnchorReactionAblation" \
  "$EXP/06.TopologyFreeMaterial" \
  "$EXP/07.ComparisonSummary"

echo "archive_dir=$ARCHIVE"
find "$EXP" -maxdepth 2 -type d | sort
