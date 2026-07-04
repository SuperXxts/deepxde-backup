PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
echo screens
screen -ls | grep bupinn_sixcase_smoke_20260628_232238 || true
echo processes
ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true
echo logs
find $PROJECT/exp/00.Smoke -maxdepth 2 -type f -name train.log -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -10
echo complete
grep -R RUN_COMPLETE $PROJECT/exp/00.Smoke/sixcase_smoke_*_20260628_232238/txt/train.log 2>/dev/null | wc -l
echo latest
latest=$(find $PROJECT/exp/00.Smoke -maxdepth 2 -type f -name train.log -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
echo $latest
[ -n "$latest" ] && grep -E 'Step|STEP|RUN_COMPLETE|Traceback|Error|Exception' "$latest" | tail -40 || true
