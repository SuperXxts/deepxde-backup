#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
DEEPXDE_ROOT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

if [ "$#" -gt 0 ]; then
  NODES=("$@")
else
  NODES=(comput1 comput2 comput3 comput4 comput5 comput6 comput7 comput8)
fi

for NODE in "${NODES[@]}"; do
  echo "===== $NODE ====="
  ssh "$NODE" "set -euo pipefail
    export PATH=/opt/dtk-24.04.3/bin:/opt/dtk-24.04/bin:/usr/local/hyhal/bin:\$PATH
    export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:/opt/dtk-24.04/lib64:/opt/dtk-24.04/lib:/opt/dtk-24.04/hip/lib:/opt/dtk-24.04/llvm/lib:\${LD_LIBRARY_PATH:-}
    source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh
    conda activate '$CONDA_ENV'
    export PYTHONPATH='$DEEPXDE_ROOT':'$PROJECT/src':\${PYTHONPATH:-}
    export DDE_BACKEND=pytorch
    echo node=\$(hostname)
    echo conda_env=\${CONDA_DEFAULT_ENV:-}
    python - <<'PY'
import os
import sys
print('python', sys.version.split()[0])
print('PYTHONPATH_head', os.environ.get('PYTHONPATH', '').split(':')[:3])
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
import deepxde as dde
print('deepxde', getattr(dde, '__version__', 'unknown'), dde.backend.backend_name)
import bupinn
print('bupinn', getattr(bupinn, '__all__', 'ok'))
PY"
done
