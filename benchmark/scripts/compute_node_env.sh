#!/bin/bash
set -euo pipefail

BASE_PATH=/opt/dtk-24.04.3/bin:/usr/local/hyhal/bin:$PATH
BASE_LD=/opt/dtk-24.04.3/lib64:/opt/dtk-24.04.3/lib:/opt/dtk-24.04.3/hip/lib:/opt/dtk-24.04.3/llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=$BASE_PATH
export LD_LIBRARY_PATH=$BASE_LD

source /public/home/xinxi/wxtian/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
conda activate pytorch2.1w2 >/dev/null 2>&1 || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
