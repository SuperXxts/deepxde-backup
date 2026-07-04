PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
echo screens
screen -ls | grep bupinn_sixcase_smoke_20260628_232238 || true
echo processes
ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true
echo complete
grep -R RUN_COMPLETE $PROJECT/exp/00.Smoke/sixcase_smoke_*_20260628_232238/txt/train.log 2>/dev/null | wc -l
echo metrics
python - <<'PY'
import glob, json, os
for f in sorted(glob.glob('/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn/exp/00.Smoke/sixcase_smoke_*_20260628_232238/metrics/metrics.json')):
    d=json.load(open(f))
    rel=d.get('relative_l2', {})
    print(os.path.basename(os.path.dirname(os.path.dirname(f))), d.get('case'), 'A=', d.get('final_A'), 'K=', rel.get('K'), 'mu=', rel.get('mu'), 'R=', d.get('reaction_rel_error'))
PY
