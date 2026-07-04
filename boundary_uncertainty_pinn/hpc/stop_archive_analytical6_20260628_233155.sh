PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
STAMP=20260628_233155
ARCHIVE_ROOT=$PROJECT/exp/01.AnalyticalBenchmark/_failed_or_stalled_${STAMP}
mkdir -p "$ARCHIVE_ROOT"
echo stop_processes
ps -u wxtian -f | grep 'mms_smooth_lens_obs225_A085_seed42_oracle_A_20260628_233155' | grep -v grep || true
pkill -f 'mms_smooth_lens_obs225_A085_seed42_oracle_A_20260628_233155' || true
echo stop_screen
screen -S bupinn_analytical6_100k_20260628_233155 -X quit || true
for d in $PROJECT/exp/01.AnalyticalBenchmark/*_${STAMP}; do
  if [ -d "$d" ]; then
    base=$(basename "$d")
    if [ ! -e "$ARCHIVE_ROOT/$base" ]; then
      mv "$d" "$ARCHIVE_ROOT/$base"
      echo archived "$d" to "$ARCHIVE_ROOT/$base"
    fi
  fi
done
echo after
ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true
screen -ls | grep bupinn_analytical6_100k_20260628_233155 || true
