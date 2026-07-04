LOG=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn/exp/01.AnalyticalBenchmark/mms_smooth_lens_obs225_A085_seed42_oracle_A_20260628_233155/txt/train.log
echo process
ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true
echo timestamp
stat -c '%y %n' "$LOG" 2>/dev/null || true
echo steps
grep -E 'Step|STEP|RUN_COMPLETE|Traceback|Error|Exception' "$LOG" 2>/dev/null | tail -80 || true
