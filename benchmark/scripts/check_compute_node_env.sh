#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/compute_node_env.sh"

if command -v hy-smi >/dev/null 2>&1; then
  echo "[check_compute_node_env] hy-smi = $(command -v hy-smi)"
  hy-smi | head -n 20 || true
fi

python - <<'PY'
import os
import torch
print('[check_compute_node_env] python =', os.sys.executable)
print('[check_compute_node_env] torch =', torch.__version__)
print('[check_compute_node_env] hip =', getattr(torch.version, 'hip', None))
print('[check_compute_node_env] cuda_available =', torch.cuda.is_available())
print('[check_compute_node_env] device_count =', torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print('[check_compute_node_env] device0 =', torch.cuda.get_device_name(0))
print('[check_compute_node_env] LD_LIBRARY_PATH =', os.environ.get('LD_LIBRARY_PATH', ''))
PY
