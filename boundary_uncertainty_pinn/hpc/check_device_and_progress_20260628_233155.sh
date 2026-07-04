PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
CUR=$PROJECT/exp/01.AnalyticalBenchmark/mms_smooth_lens_obs225_A085_seed42_oracle_A_20260628_233155
SMOKE=$PROJECT/exp/00.Smoke/smoke_mms_pfnn_A_seed42_
echo process
ps -u wxtian -o pid,ppid,pcpu,pmem,etime,cmd | grep train_deepxde_pfnn.py | grep -v grep || true
echo log_stat
stat -c '%y %s %n' "$CUR/txt/train.log" 2>/dev/null || true
echo log_tail
tail -80 "$CUR/txt/train.log" 2>/dev/null || true
echo smoke_config_device
python - <<'PY'
import json
p='/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn/exp/00.Smoke/smoke_mms_pfnn_A_seed42_/json/config.json'
d=json.load(open(p))
print('backend=', d.get('backend'), 'cuda_available=', d.get('torch_cuda_available'), 'device_count=', d.get('torch_device_count'))
PY
echo gpu_probe
(command -v rocm-smi >/dev/null && rocm-smi --showuse --showmemuse) || (command -v hy-smi >/dev/null && hy-smi) || (command -v nvidia-smi >/dev/null && nvidia-smi) || true
