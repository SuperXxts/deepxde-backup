#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-pytorch2.1w2}

echo "node=$(hostname)"
echo "user=$(whoami)"
echo "---- dtk dirs ----"
ls -ld /opt/dtk-24.04.3 /opt/dtk-24.04 /usr/local/hyhal 2>/dev/null || true
echo "---- needed libs ----"
find /opt /usr/local -name libgalaxyhip.so.5 -o -name libamd_comgr.so -o -name libamd_comgr.so.2 2>/dev/null | sort | sed -n '1,160p' || true
echo "---- command probes ----"
command -v hy-smi || true
command -v rocm-smi || true
command -v hipcc || true
command -v python || true

source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"
echo "python=$(command -v python)"
echo "---- torch import with candidate LD_LIBRARY_PATH ----"
export PATH=/opt/dtk-24.04.3/bin:/opt/dtk-24.04/bin:/usr/local/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:/opt/dtk-24.04/lib64:/opt/dtk-24.04/lib:/opt/dtk-24.04/hip/lib:/opt/dtk-24.04/llvm/lib:${LD_LIBRARY_PATH:-}
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python - <<'PY'
import os
import sys
print("python_version", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("device", i, torch.cuda.get_device_name(i))
except Exception as exc:
    print("TORCH_IMPORT_ERROR", repr(exc))
    raise
PY
