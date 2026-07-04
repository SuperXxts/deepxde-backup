PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
echo screens
screen -ls | grep bupinn_analytical6_100k_20260628_233155 || true
echo processes
ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true
echo logs
find $PROJECT/exp/01.AnalyticalBenchmark -maxdepth 2 -type f -name train.log -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -10
echo latest
latest=$(find $PROJECT/exp/01.AnalyticalBenchmark -maxdepth 2 -type f -name train.log -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
echo $latest
[ -n "$latest" ] && grep -E 'Step|STEP|RUN_COMPLETE|Traceback|Error|Exception' "$latest" | tail -50 || true
echo complete
grep -R RUN_COMPLETE $PROJECT/exp/01.AnalyticalBenchmark/*_20260628_233155/txt/train.log 2>/dev/null | wc -l
